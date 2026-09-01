// Qwen3.8-27B DeltaNet decode 单步融合 kernel (路线 R2, v2)
// ============================================================
// 融合 48 层 linear_attention 在 T=1 时的全部中间算子:
//   causal_conv1d_update + silu + q/k l2norm + gated delta rule recurrent
//   + RMSNormGated —— 约 100 次 aten op/层 (含 ~50 次 dtype 转换) → 2 次 launch/层
//
// v2 修复 (对照 transformers 5.14.1 modeling_qwen3_5.py 逐行审阅):
//   1. conv_state 4 帧滚动 (torch_causal_conv1d_update L224: state 存最近
//      kernel_size=4 帧, conv 窗口 = [s1,s2,s3,x], s0 丢弃; v1 误为 3 帧)
//   2. 拆 2 个 kernel: conv 区 scratch 存在跨 block 读写依赖
//      (block h 的 Phase B 要读 block 0..31 写的 q/k 区), __syncthreads
//      无法跨 block 同步 → kernel1 写完 scratch, kernel2 消费 (stream 顺序保证)
//   3. fp16 舍入对齐: 参考路径 conv_out/q/k/v/o 均经 .to(fp16) 舍入后再
//      转 fp32 计算 (L238/L336/L369), v1 全 fp32 会累积偏差 → 补 __float2half
//   4. beta 补 fp16 舍入 (b.sigmoid() 在 fp16 tensor 上, L520)
//
// 语义对照 (纯 torch 回退):
//   torch_causal_conv1d_update (L224):
//     in = cat(state[4], x) → conv1d(k=4) 取末帧 → silu; state ← [s1,s2,s3,x]
//   torch_recurrent_gated_delta_rule (L329) 全程 fp32:
//     g_t=exp(g); S*=g_t; kv[j]=Σ_i S[i][j]·k[i]; δ=(v-kv)·β;
//     S[i][j]+=k[i]·δ[j]; o[j]=Σ_i q[i]·S_new[i][j]
//     q,k 先 l2norm(eps=1e-6); q 再 × 1/sqrt(128)
//   Qwen3_5RMSNormGated (L189):
//     var=mean(x²)(fp32); xn=x·rsqrt(var+eps); xn→fp16 × w(fp16);
//     × silu(z fp32) → fp16
//   标量 (L520-522): β=sigmoid(b fp16); g=-exp(A_log)·softplus(a+dt_bias) fp32
//
// recurrent kernel 布局: 48 blocks (每 block 一个 v-head) × 128 threads (v-dim j)
//   recurrent 状态存转置 S_t[h][j][i] —— thread j 独占连续一行 128 float,
//   两次遍历均在寄存器/L1 完成, 无跨线程依赖
//   v_head h ↔ k_head h/3 (num_v/num_k = 48/16 = 3, 对齐 repeat_interleave)
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>

#define DK        128     // head_k_dim = head_v_dim
#define NVH       48      // linear_num_value_heads
#define KEY_DIM   2048    // 16 heads × 128
#define VAL_DIM   6144    // 48 heads × 128
#define CONV_DIM  10240   // 2×2048 + 6144
#define NTHREADS  128

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float softplus_f(float x) {
    return (x > 20.0f) ? x : log1pf(expf(x));
}
// fp32 计算后舍入到 fp16 再转回 —— 模拟 torch half tensor 的存取舍入
__device__ __forceinline__ float round_hf(float x) {
    return __half2float(__float2half(x));
}

// ---- kernel 1: conv1d update + silu (10240 通道并行) ----
__global__ void dn_conv_kernel(
    const __half* __restrict__ mixed,      // [10240] in_proj_qkv 输出 (fp16)
    const __half* __restrict__ conv_w,     // [10240*4] conv1d.weight (fp16)
    __half*       __restrict__ conv_state, // [10240*4] in/out, 布局 c*4+t
    float*        __restrict__ scratch)    // [10240] conv+silu 结果 (含 fp16 舍入)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= CONV_DIM) return;
    const __half s0 = conv_state[c*4+0], s1 = conv_state[c*4+1],
                 s2 = conv_state[c*4+2], s3 = conv_state[c*4+3];
    const __half x  = mixed[c];
    // conv 窗口 [s1,s2,s3,x] (cross-correlation, 参考 L236 out1)
    float acc = __half2float(conv_w[c*4+0]) * __half2float(s1)
              + __half2float(conv_w[c*4+1]) * __half2float(s2)
              + __half2float(conv_w[c*4+2]) * __half2float(s3)
              + __half2float(conv_w[c*4+3]) * __half2float(x);
    scratch[c] = round_hf(silu_f(acc));       // 参考: silu 后 out.to(fp16)
    conv_state[c*4+0] = s1;                   // state ← [s1,s2,s3,x]
    conv_state[c*4+1] = s2;
    conv_state[c*4+2] = s3;
    conv_state[c*4+3] = x;
}

// 128 线程 (4 warps) 块内求和
__device__ __forceinline__ float block_sum(float v, float* red) {
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5;
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    if (lane == 0) red[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < 4) ? red[lane] : 0.0f;
        v += __shfl_down_sync(0xffffffffu, v, 2);
        v += __shfl_down_sync(0xffffffffu, v, 1);
        if (lane == 0) red[4] = v;
    }
    __syncthreads();
    return red[4];
}

// ---- kernel 2: 标量 + l2norm + recurrent + RMSNormGated (每 block 一个 v-head) ----
__global__ void __launch_bounds__(NTHREADS) dn_rec_kernel(
    const __half* __restrict__ z_in,       // [6144]  in_proj_z 输出
    const __half* __restrict__ b_in,       // [48]    in_proj_b 输出
    const __half* __restrict__ a_in,       // [48]    in_proj_a 输出
    const __half* __restrict__ A_log,      // [48]
    const __half* __restrict__ dt_bias,    // [48]
    const __half* __restrict__ norm_w,     // [128] RMSNormGated.weight
    const float*  __restrict__ scratch,    // [10240] conv 结果 (kernel 1 写入)
    float*        __restrict__ state,      // [48*128*128] S_t (转置), in/out
    __half*       __restrict__ out,        // [6144] 送 out_proj
    float rms_eps)
{
    const int h = blockIdx.x;                       // v-head
    const int j = threadIdx.x;                      // v-dim
    __shared__ float s_q[DK], s_k[DK], red[8];

    // ---- 标量 (参考 L520-522) ----
    const int   kh   = h / 3;                       // k-head (repeat_interleave 3)
    const float beta = round_hf(1.0f / (1.0f + expf(-__half2float(b_in[h]))));
    const float g    = -expf(__half2float(A_log[h]))
                          * softplus_f(__half2float(a_in[h]) + __half2float(dt_bias[h]));
    const float eg   = expf(g);

    // conv 后布局 [q 2048 | k 2048 | v 6144] (参考 L506 split)
    s_q[j] = scratch[kh * DK + j];
    s_k[j] = scratch[KEY_DIM + kh * DK + j];
    __syncthreads();

    // l2norm (eps 1e-6) + q 缩放 1/sqrt(128)  (参考 L242/L334/L343)
    const float inv_q = rsqrtf(block_sum(s_q[j] * s_q[j], red) + 1e-6f)
                        * 0.08838834764831845f;
    const float inv_k = rsqrtf(block_sum(s_k[j] * s_k[j], red) + 1e-6f);
    s_q[j] *= inv_q;
    s_k[j] *= inv_k;
    __syncthreads();

    // ---- recurrent gated delta rule (fp32, 参考 L354-365) ----
    const float v_j = scratch[KEY_DIM * 2 + h * DK + j];
    float* Srow = state + (size_t)(h * DK + j) * DK;   // S_t[h][j][i], j=v 行
    float kv = 0.0f;
    for (int i = 0; i < DK; i++) kv += (Srow[i] * eg) * s_k[i];
    const float delta = (v_j - kv) * beta;
    float o = 0.0f;
    for (int i = 0; i < DK; i++) {
        const float s = Srow[i] * eg + s_k[i] * delta;
        Srow[i] = s;
        o += s_q[i] * s;
    }
    o = round_hf(o);        // 参考 L369 core_attn_out.to(fp16) 后 norm 再转 fp32

    // ---- RMSNormGated (参考 L195-204) ----
    const float var_mean = block_sum(o * o, red) / DK;
    const float xn  = o * rsqrtf(var_mean + rms_eps);
    const __half hw = __hmul(__float2half(xn), norm_w[j]);   // fp16 乘, 对齐 L201
    const float res = __half2float(hw) * silu_f(__half2float(z_in[h * DK + j]));
    out[h * DK + j] = __float2half(res);
}

extern "C" {

__declspec(dllexport) int launch_dn_fused(
    const void* mixed, const void* z_in, const void* b_in, const void* a_in,
    const void* conv_w, void* conv_state, const void* A_log, const void* dt_bias,
    const void* norm_w, void* state, void* scratch, void* out,
    float rms_eps, void* stream)
{
    cudaStream_t s = (cudaStream_t)stream;
    dn_conv_kernel<<<(CONV_DIM + 255) / 256, 256, 0, s>>>(
        (const __half*)mixed, (const __half*)conv_w, (__half*)conv_state,
        (float*)scratch);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "dn_conv launch: %s\n", cudaGetErrorString(e)); return -1; }

    dn_rec_kernel<<<NVH, NTHREADS, 0, s>>>(
        (const __half*)z_in, (const __half*)b_in, (const __half*)a_in,
        (const __half*)A_log, (const __half*)dt_bias, (const __half*)norm_w,
        (const float*)scratch, (float*)state, (__half*)out, rms_eps);
    e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "dn_rec launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

}   // extern "C"
