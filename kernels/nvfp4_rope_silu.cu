// nvfp4_rope_silu.cu — P9: RoPE QK + MLP silu·mul 融合 kernel (sm_70)
// ====================================================================
// 目标: 消除 decode 每步 elementwise 残余 (P6 修正分布 ~2.0ms/步):
//   1. rope_qk_kernel: 16 层 full_attention 的 apply_rotary_pos_emb
//      (q 24×256 + k 4×256, torch 链 ~10 kernel/层 ~0.8ms/步)
//      → 1 launch/层
//   2. silu_mul_kernel: 64 层 MLP 的 silu(gate)*up (2 kernel/层 ~0.5ms/步)
//      → 1 launch/层
//
// 位精确舍入链 (逐 op 对齐 torch CUDA opmath=fp32):
//   RoPE: t1=f16(q·cos) t2=f16(rot(q)·sin) out=f16(t1+t2)
//     rotate_half=(-x2,x1) 符号翻转位精确; 每步 fp32 计算后舍入 f16
//   silu: torch 实现 x/(1+exp(-x)) — 必须用除法 (非乘倒数, ulp 差异)
//     s=f16(g/(1+expf(-g))) out=f16(f32(s)·f32(u))
//   注: expf 用软件精确版, 禁用 __expf (快速数学不位对齐, 发现64教训)
//
// 用法: 由 _qwen38_p9_fuse.py patch (T=1 专属, T>1 回退原路径)

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define D   256
#define DH  128

// ---- RoPE: grid=(hq+hkv) blocks × 256 threads, 每 block 一个头 ----
__global__ void rope_qk_kernel(
    const __half* __restrict__ q,     // [hq*256] 每头连续
    const __half* __restrict__ k,     // [hkv*256]
    const __half* __restrict__ cs,    // [256] fp16 cos
    const __half* __restrict__ sn,    // [256] fp16 sin
    __half* __restrict__ q_out,
    __half* __restrict__ k_out,
    int hq)
{
    const int h = blockIdx.x;              // [0, hq+hkv)
    const int d = threadIdx.x;             // [0, 256)
    const bool is_q = h < hq;
    const size_t off = (size_t)(is_q ? h : h - hq) * D;
    const __half* src = is_q ? q + off : k + off;
    __half* dst = is_q ? q_out + off : k_out + off;

    const float v  = __half2float(src[d]);
    const float c  = __half2float(cs[d]);
    const float s  = __half2float(sn[d]);
    const float vp = __half2float(src[(d < DH) ? d + DH : d - DH]);
    const float vpp = (d < DH) ? -vp : vp;     // rotate_half: (-x2, x1)

    const __half t1 = __float2half(v * c);
    const __half t2 = __float2half(vpp * s);
    dst[d] = __float2half(__half2float(t1) + __half2float(t2));
}

// ---- silu·mul: grid=ceil(n/256) × 256 threads ----
__global__ void silu_mul_kernel(
    const __half* __restrict__ g,     // [n] gate_proj 输出
    const __half* __restrict__ u,     // [n] up_proj 输出
    __half* __restrict__ out,          // [n]
    int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gf = __half2float(g[i]);
    const float uf = __half2float(u[i]);
    const __half s = __float2half(gf / (1.0f + expf(-gf)));  // torch: x/(1+exp(-x))
    out[i] = __float2half(__half2float(s) * uf);
}

extern "C" {
__declspec(dllexport)
int launch_rope_qk(const void* q, const void* k, const void* cs, const void* sn,
                   void* q_out, void* k_out, int hq, int hkv, void* stream)
{
    if (hq + hkv <= 0) return -1;
    cudaStream_t st = (cudaStream_t)stream;
    rope_qk_kernel<<<hq + hkv, 256, 0, st>>>(
        (const __half*)q, (const __half*)k, (const __half*)cs, (const __half*)sn,
        (__half*)q_out, (__half*)k_out, hq);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

__declspec(dllexport)
int launch_silu_mul(const void* g, const void* u, void* out, int n, void* stream)
{
    if (n <= 0) return -1;
    const int blocks = (n + 255) / 256;
    cudaStream_t st = (cudaStream_t)stream;
    silu_mul_kernel<<<blocks, 256, 0, st>>>(
        (const __half*)g, (const __half*)u, (__half*)out, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
}
