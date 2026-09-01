#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Qwen3.8-27B RMSNorm 融合 kernel (P5-A-2)
// ============================================================
// Qwen3_5RMSNorm (modeling_qwen3_5.py L749-754) 在 T=1 decode 时
// 每处展开 10 个 torch 小内核 (含每步重算 w.float() 和 1.0+w):
//   copy(x.float) + pow + mean-reduce + add(var+eps) + rsqrt
//   + mul + copy(w.float) + add(1+w) + mul + copy(type_as)
// 全模型 161 处/步 (64层×2 + final norm + 16×q/k_norm) ≈ 5.5ms/步。
// 本 kernel 每处 1 个 launch (~3μs)。
//
// 语义 (zero-centered weight, Qwen3.5 风格 (x*w).to(f16)):
//   out = f16( f32(x) · rsqrt(mean(x²)+eps) · (1+w_f32) )
// w1 = (1.0 + w.float()) 在 patch 时预计算一次 (权重静态)。
// 舍入链位对齐 torch: x.float() 无舍入; mean 归约顺序差异 ~1ulp;
// rsqrt/mul 均 fp32; 仅末尾一次 f16 舍入 (type_as)。
//
// 布局: x 行主序但允许行间 gap (chunk view: stride(-2)=512, n=256),
//   row r 起点 = x + r*row_stride; out 连续 [rows*n]。
// grid = rows (每 block 一行), block = n>=1024 ? 1024 : 256。

extern "C" __global__ void rmsnorm_kernel(
    const __half* __restrict__ x,      // [rows*row_stride] 行可有 gap
    const long long  row_stride,       // 行间元素数
    const float*   __restrict__ w1,    // [n] 预计算 1+w (fp32)
    __half*        __restrict__ out,   // [rows*n] 连续
    int n, float eps)
{
    const int r = blockIdx.x;
    const int t = threadIdx.x;
    const __half* xr = x + (size_t)r * (size_t)row_stride;
    __shared__ float s_red[32];

    // ---- Σx² (fp32) ----
    float ss = 0.0f;
    for (int i = t; i < n; i += blockDim.x) {
        const float v = __half2float(xr[i]);
        ss += v * v;
    }
    const int warp = t >> 5, lane = t & 31;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        ss += __shfl_down_sync(0xffffffffu, ss, o);
    if (lane == 0) s_red[warp] = ss;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < (int)(blockDim.x >> 5)) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();

    // ---- out = x · rsqrt(mean+eps) · (1+w) ----
    const float inv = rsqrtf(s_red[0] / (float)n + eps);
    __half* orow = out + (size_t)r * n;
    for (int i = t; i < n; i += blockDim.x)
        orow[i] = __float2half(__half2float(xr[i]) * inv * w1[i]);
}

// host 入口 (ctypes)
extern "C" __declspec(dllexport) int launch_rmsnorm(
    const void* x, long long row_stride, const void* w1, void* out,
    int rows, int n, float eps, void* stream)
{
    if (rows <= 0 || n <= 0) return -1;
    const int block = (n >= 1024) ? 1024 : 256;
    rmsnorm_kernel<<<rows, block, 0, (cudaStream_t)stream>>>(
        (const __half*)x, row_stride, (const float*)w1,
        (__half*)out, n, eps);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
