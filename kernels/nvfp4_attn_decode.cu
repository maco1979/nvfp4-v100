#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>

// Qwen3.8-27B attention decode kernel (P5-B → P6 v3.1)
// ============================================================
// v3.1 修正: exp2f 回退 expf。exp2((s·scale·log2e)-m) 与 expf(s·scale-m)
//   的 ulp 差异经 63 步贪心解码放大, 导致 token 逐位验收 FAIL
//   (首 8 token 一致, 后续发散)。按"精度优先于速度"回退。
// v3 保留的两项优化 (均位安全):
//   1. p 预计算: v2 Pass2 每 thread 重算 exp(s-m)·inv_l
//      (L×256 次 exp) → reduce 后并行算一遍 p[i] 原地写回
//      s_sc (L 次), Pass2 变纯乘加。每 thread 计算的是同一
//      表达式, 值位相同。
//   2. Pass1 half2 加载: k 行点积 load 指令减半 (256→128),
//      half2→float2 转换位精确, fp32 累加保精度。
//
// 语义不变: score=(q·k)·scale → softmax fp32 → Σp·v → gate·σ 融合。

#ifndef MAXLEN
#define MAXLEN 3072
#endif
#define D       256
#define DH2     (D / 2)   // 128 half2
#define HQ      24
#define HKV     4
#define GQA     6
#define NTHREADS 256

extern "C" __global__ void __launch_bounds__(NTHREADS) attn_decode_kernel(
    const __half* __restrict__ q,        // [24*256]
    const __half* __restrict__ keys,    // [4*MAXLEN*256]
    const __half* __restrict__ values,  // [4*MAXLEN*256]
    const long long* __restrict__ cum_len,  // device int64, = L (含当前 token)
    const __half* __restrict__ gate,   // [24*256] fp16 门控半区
    __half* __restrict__ out,           // [24*256]
    float scale)
{
    const int h = blockIdx.x;
    const int kv = h / GQA;
    const int t = threadIdx.x;
    const int L = (int)(*cum_len);

    __shared__ __half s_q[D];
    __shared__ float s_sc[MAXLEN];
    __shared__ float s_red[NTHREADS / 32];

    s_q[t] = q[h * D + t];

    // ---- Pass1: scores = (q·k_i)·scale (half2 load, fp32 累加) ----
    const __half2* kbase2 = (const __half2*)(keys + (size_t)kv * MAXLEN * D);
    const __half2* q2 = (const __half2*)s_q;
    for (int i = t; i < L; i += NTHREADS) {
        const __half2* ki = kbase2 + (size_t)i * DH2;
        float acc = 0.0f;
        #pragma unroll 32
        for (int d = 0; d < DH2; d++) {
            const float2 kk = __half22float2(ki[d]);
            const float2 qq = __half22float2(q2[d]);
            acc = fmaf(qq.x, kk.x, acc);
            acc = fmaf(qq.y, kk.y, acc);
        }
        s_sc[i] = acc * scale;
    }
    __syncthreads();

    // ---- block reduce: m = max, l = Σ exp(s - m) ----
    const int warp = t >> 5, lane = t & 31;
    float m = -3.4e38f;
    for (int i = t; i < L; i += NTHREADS) m = fmaxf(m, s_sc[i]);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        m = fmaxf(m, __shfl_down_sync(0xffffffffu, m, o));
    if (lane == 0) s_red[warp] = m;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < NTHREADS / 32) ? s_red[lane] : -3.4e38f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1)
            v = fmaxf(v, __shfl_down_sync(0xffffffffu, v, o));
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    m = s_red[0];

    float l = 0.0f;
    for (int i = t; i < L; i += NTHREADS) l += expf(s_sc[i] - m);
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        l += __shfl_down_sync(0xffffffffu, l, o);
    if (lane == 0) s_red[warp] = l;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < NTHREADS / 32) ? s_red[lane] : 0.0f;
        #pragma unroll
        for (int o = 4; o > 0; o >>= 1)
            v += __shfl_down_sync(0xffffffffu, v, o);
        if (lane == 0) s_red[0] = v;
    }
    __syncthreads();
    const float inv_l = 1.0f / s_red[0];

    // ---- p 预计算: s_sc[i] ← exp(s_i - m)·inv_l (L 次, 原地) ----
    for (int i = t; i < L; i += NTHREADS)
        s_sc[i] = expf(s_sc[i] - m) * inv_l;
    __syncthreads();

    // ---- Pass2: o_d = Σ_i p_i · v_i[d] (纯乘加, 无 exp) ----
    const __half* vbase = values + (size_t)kv * MAXLEN * D;
    float acc = 0.0f;
    for (int i = 0; i < L; i++)
        acc = fmaf(s_sc[i], __half2float(vbase[(size_t)i * D + t]), acc);

    // gate·σ 融合 (位精确复刻 torch 舍入链)
    const float a = __half2float(__float2half(acc));
    const float g = __half2float(gate[h * D + t]);
    const float s = __half2float(__float2half(1.0f / (1.0f + expf(-g))));
    out[h * D + t] = __float2half(a * s);
}

extern "C" __declspec(dllexport) int launch_attn_decode(
    const void* q, const void* keys, const void* values,
    const void* cum_len, const void* gate, void* out,
    int max_len, float scale, void* stream)
{
    if (max_len != MAXLEN) return -2;
    attn_decode_kernel<<<HQ, NTHREADS, 0, (cudaStream_t)stream>>>(
        (const __half*)q, (const __half*)keys, (const __half*)values,
        (const long long*)cum_len, (const __half*)gate, (__half*)out, scale);
    cudaError_t e = cudaGetLastError();
    return (e == cudaSuccess) ? 0 : (int)e;
}
