// NVFP4 GEMV kernel — decode 场景 (T=1) 专用 + GEMV2 (T=2, MTP verify)
// ============================================================
// output[M] = weight[M,K] @ act[K]
// weight: packed (M, K/2) uint8 + scales (M, K/16) uint8 (v2 符号内嵌布局)
//
// v3 (2026-08-17): byte->half2 LUT 反查表 — 解 INT 计算瓶颈
//   v2 实测 305-392GB/s。分析: 每字节权重 (2 元素) 需 ~14 次 INT 位操作
//   (nibble 提取x2 + e2m1 位构造x2 + 符号拼装 + half 乘 scale),
//   V100 INT 吞吐 ~64-128 op/cycle/SM -> 反量化算力先于带宽饱和。
//   v3 改 256 项 byte->half2 共享内存 LUT (1KB, 块首一次性构建):
//     每字节 = 1 次 LDS 4B + 1 次 __hfma2, 位操作清零
//     scale 改块尾 float 结算 (E8M0->float 纯位构造 sb<<23, 发现40)
//   实测 v3.2 (LUT + launch_bounds 锁 8 blocks/SM): 419-518 GB/s (+37%)
// 中文注释文件头带 BOM (MSVC GBK 陷阱, 发现: v3 越界修复期)
#include <cuda_fp16.h>
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define NVFP4_BLOCK_SIZE 16
#define E8M0_BIAS 127
#define WARPS_PER_BLOCK 8
#define GEMV_THREADS 256

// ---- 剖析探针宏 (D1, 关态零影响): 对照实验法定位 T=1 瓶颈分量 ----
// LUT_BYPASS: LDS 查表换常量 (结果错, 只测速) -> LDS/bank-conflict 成本
// WBYPASS:   权重+scales 全局读换常量 -> DRAM 权重流成本 (compute floor)
// ACT_BYPASS: 激活读换常量 -> act L1/L2 成本
#ifdef LUT_BYPASS
#define LUT_LOOKUP(lut, byte) ((lut)[0])
#else
#define LUT_LOOKUP(lut, byte) ((lut)[byte])
#endif

// E2M1 幅值码点 -> half bits (纯位构造, v5.2 发现27) — 仅 LUT 构建期使用
//   code>=2: 0x3800 + (code<<9); code==1: 0.5->0x3800; code==0: 0
__host__ __device__ __forceinline__ uint32_t e2m1_to_half_bits(uint32_t code) {
    return (code >= 2u) ? (0x3800u + (code << 9))
         : (code == 1u) ? 0x3800u : 0u;
}

// E8M0 scale byte -> float (纯位构造): 2^(sb-127) 的 float bits = sb<<23
//   (float 指数域 bias 127, E8M0 也是 bias 127 -> 指数域直通, 发现40)
__device__ __forceinline__ float e8m0_to_float(uint32_t sb) {
    return __int_as_float((int)(sb << 23));
}

// LUT 构建: 256 项 byte -> half2 (低 nibble = low half), 全块共用
__device__ __forceinline__ void build_lut(uint32_t* lut, uint32_t tid) {
    uint32_t b = tid;                               // 256 线程各建 1 项
    uint32_t n0 = b & 0xFu, n1 = b >> 4;
    uint32_t h0 = e2m1_to_half_bits(n0 & 7u) | ((n0 & 8u) << 12);
    uint32_t h1 = e2m1_to_half_bits(n1 & 7u) | ((n1 & 8u) << 12);
    lut[b] = (h1 << 16) | h0;
}

// ---- GEMV (T=1): 每 warp 一行, 每 lane 每迭代 2 block (16B packed) ----
__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K]
    __half*        __restrict__ output,          // [M]
    int M, int K)
{
    __shared__ uint32_t lut[256];
    build_lut(lut, threadIdx.x);
    __syncthreads();

    const int nblocks = K >> 4;                  // K/16
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc = 0.0f;
    __half2 acc2 = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
#ifdef WBYPASS                      // 探针: 权重流换常量
        uint4 pk = make_uint4(0x01234567u, 0x89ABCDEFu, 0x02468ACEu, 0x13579BDFu);
        uint32_t sc2 = 0x7F7Fu;
#else
        uint4 pk = __ldg(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldg(reinterpret_cast<const uint16_t*>(row_sc + b));
#endif

#ifdef ACT_BYPASS                   // 探针: act 读换常量
        uint4 aa[4] = {make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u),
                       make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u),
                       make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u),
                       make_uint4(0x3C003C00u, 0x3C003C00u, 0x3C003C00u, 0x3C003C00u)};
#else
        const uint4* a4 = reinterpret_cast<const uint4*>(activation + (size_t)(b << 4));
        uint4 aa[4] = {__ldg(a4), __ldg(a4 + 1), __ldg(a4 + 2), __ldg(a4 + 3)};
#endif

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {            // j=0,1 block_a; j=2,3 block_b
            const uint32_t w = wq[j];
            uint32_t sc = (j < 2) ? (sc2 & 0xFFu) : (sc2 >> 8);
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t byte = (w >> (8 * p)) & 0xFFu;
                __half2 wv = *reinterpret_cast<const __half2*>(
                    &LUT_LOOKUP(lut, byte));
                __half2 avv = *reinterpret_cast<const __half2*>(
                    &aa[2 * (j >> 1)].x + (j & 1) * 4 + p);
                acc2 = __hfma2(wv, avv, acc2);
            }
            // 块尾结算: half2 和 x E8M0 scale
            acc += (__low2float(acc2) + __high2float(acc2)) * e8m0_to_float(sc);
            acc2 = __float2half2_rn(0.0f);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) output[warp_global] = __float2half(acc);
}

// ---- GEMV-NL (T=1, v4 no-LUT): 位构造反量化, 0 LDS ----
// D1 剖析 (对照实验, M=31232): 基线 191.7us; 去 LUT-LDS 108.4us (-43%)
//   -> LDS 随机 bank conflict (birthday ~1.54x 串行) 是 T=1 首瓶颈, DRAM 未饱和。
//   GEMV2-NL (T=2) 已证位构造可行 (发现41); 本内核 = 同思路 T=1 版。
// 数学恒等: 位构造产 half 与 LUT 完全相同 -> 输出 bit-exact, 验收即回测。
__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemvnl_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K]
    __half*        __restrict__ output,          // [M]
    int M, int K)
{
    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc = 0.0f;
    __half2 acc2 = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
        uint4 pk = __ldg(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldg(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a4 = reinterpret_cast<const uint4*>(activation + (size_t)(b << 4));
        uint4 aa[4] = {__ldg(a4), __ldg(a4 + 1), __ldg(a4 + 2), __ldg(a4 + 3)};

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t w = wq[j];
            uint32_t sc = (j < 2) ? (sc2 & 0xFFu) : (sc2 >> 8);
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t byte = (w >> (8 * p)) & 0xFFu;
                uint32_t n0 = byte & 0xFu, n1 = byte >> 4;
                uint32_t h0 = e2m1_to_half_bits(n0 & 7u) | ((n0 & 8u) << 12);
                uint32_t h1 = e2m1_to_half_bits(n1 & 7u) | ((n1 & 8u) << 12);
                uint32_t wpack = h0 | (h1 << 16);
                __half2 wv = *reinterpret_cast<const __half2*>(&wpack);
                __half2 avv = *reinterpret_cast<const __half2*>(
                    &aa[2 * (j >> 1)].x + (j & 1) * 4 + p);
                acc2 = __hfma2(wv, avv, acc2);
            }
            acc += (__low2float(acc2) + __high2float(acc2)) * e8m0_to_float(sc);
            acc2 = __float2half2_rn(0.0f);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) output[warp_global] = __float2half(acc);
}

// ---- GEMV-V4 (T=1, LDS 隐藏 + L2 分流): 三项叠加 ----
// 1) 批量 LDS: 16 查表前置独立发射 (v3.4 T=2 同款, 冲突串行段与 hfma2 重叠)
// 2) 双 acc2 链: 打破 16 连 hfma2 的串行依赖 (ILP)
// 3) 权重/scales __ldcs (evict-first): 权重流零复用, 让 act 常驻 L2
//    (p3 探针: act 读占 40% 运行时 — 疑似被权重流冲出 L2)
__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv_v4_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K]
    __half*        __restrict__ output,          // [M]
    int M, int K)
{
    __shared__ uint32_t lut[256];
    build_lut(lut, threadIdx.x);
    __syncthreads();

    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc = 0.0f;
    __half2 accA = __float2half2_rn(0.0f), accB = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
        uint4 pk = __ldcs(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldcs(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a4 = reinterpret_cast<const uint4*>(activation + (size_t)(b << 4));
        uint4 aa[4] = {__ldg(a4), __ldg(a4 + 1), __ldg(a4 + 2), __ldg(a4 + 3)};

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        __half2 wv[16];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int p = 0; p < 4; p++)
                wv[j * 4 + p] = *reinterpret_cast<const __half2*>(
                    &LUT_LOOKUP(lut, (wq[j] >> (8 * p)) & 0xFFu));

        // 双链交错: j=0,1 -> accA, j=2,3 -> accB (每链 8 连 hfma2 -> 4+4)
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                __half2 avA = *reinterpret_cast<const __half2*>(
                    &aa[j].x + p);
                __half2 avB = *reinterpret_cast<const __half2*>(
                    &aa[2 + j].x + p);
                accA = __hfma2(wv[j * 4 + p], avA, accA);
                accB = __hfma2(wv[(j + 2) * 4 + p], avB, accB);
            }
        }
        acc += (__low2float(accA) + __high2float(accA)) * e8m0_to_float(sc2 & 0xFFu)
             + (__low2float(accB) + __high2float(accB)) * e8m0_to_float(sc2 >> 8);
        accA = __float2half2_rn(0.0f);
        accB = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) output[warp_global] = __float2half(acc);
}

// ---- GEMV-V5 (T=1, global-LUT): 1KB 表驻全局, __ldg 走 L1-TEX 通路 ----
// 依据: p1 探针 LDS->常量 = 831 GB/s (LDS 随机冲突 43%); p4 位构造 = 395 (ALU 瓶颈);
//       v4 批量+ldcs = 467 (冲突在管线内串行, 发射提前无效)。
// L1-TEX 随机 4B 命中与 LDS bank 机制不同 — 唯一未试的 LUT 通路。
__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv_v5_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K]
    __half*        __restrict__ output,          // [M]
    const uint32_t* __restrict__ glut,           // [256] 全局 1KB 表
    int M, int K)
{
    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc = 0.0f;
    __half2 accA = __float2half2_rn(0.0f), accB = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
        uint4 pk = __ldcs(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldcs(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a4 = reinterpret_cast<const uint4*>(activation + (size_t)(b << 4));
        uint4 aa[4] = {__ldg(a4), __ldg(a4 + 1), __ldg(a4 + 2), __ldg(a4 + 3)};

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        __half2 wv[16];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t lv = __ldg(glut + ((wq[j] >> (8 * p)) & 0xFFu));
                wv[j * 4 + p] = *reinterpret_cast<const __half2*>(&lv);
            }

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                __half2 avA = *reinterpret_cast<const __half2*>(&aa[j].x + p);
                __half2 avB = *reinterpret_cast<const __half2*>(&aa[2 + j].x + p);
                accA = __hfma2(wv[j * 4 + p], avA, accA);
                accB = __hfma2(wv[(j + 2) * 4 + p], avB, accB);
            }
        }
        acc += (__low2float(accA) + __high2float(accA)) * e8m0_to_float(sc2 & 0xFFu)
             + (__low2float(accB) + __high2float(accB)) * e8m0_to_float(sc2 >> 8);
        accA = __float2half2_rn(0.0f);
        accB = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) output[warp_global] = __float2half(acc);
}

// ---- GEMV-V6 (T=1, 4相 LUT): 根除 LDS 随机 bank conflict ----
// 冲突模型修正: 每条标量 LDS 指令是 warp 级 — 32 lane 各查随机 idx,
//   32 球落 32 bank, 期望冲突链 ~3-4 (与 p1 探针 43% LDS 损失吻合)。
// 对策: LUT 放 4 份相位偏移副本 (起点偏移 8 项 = bank 相位差 8),
//   lane 按 (lane>>3) 选相位 -> 32 随机地址分 4 组, 组间 bank 相位互错,
//   期望冲突链 ~3.5 降至 ~1.5。重叠区内容天然一致 (同一张表)。
//   循环外 1 次指针计算, 循环内零额外指令。
// 表大小: 256 + 3*8 = 280 项 (1120B)
__device__ __forceinline__ void build_lut4(uint32_t* lut, uint32_t tid) {
    // 280 项, 256 线程各建 1 项 + 前 24 项再建一次
    uint32_t idx = tid;
    uint32_t n0 = idx & 0xFu, n1 = (idx >> 4) & 0xFu;
    uint32_t h0 = e2m1_to_half_bits(n0 & 7u) | ((n0 & 8u) << 12);
    uint32_t h1 = e2m1_to_half_bits(n1 & 7u) | ((n1 & 8u) << 12);
    lut[idx] = (h1 << 16) | h0;
    idx = tid + 256;                                    // 补 24 项尾部
    if (idx < 280) {
        uint32_t m0 = idx & 0xFu, m1 = (idx >> 4) & 0xFu;
        uint32_t g0 = e2m1_to_half_bits(m0 & 7u) | ((m0 & 8u) << 12);
        uint32_t g1 = e2m1_to_half_bits(m1 & 7u) | ((m1 & 8u) << 12);
        lut[idx] = (g1 << 16) | g0;
    }
}

__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv_v6_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K]
    __half*        __restrict__ output,          // [M]
    int M, int K)
{
    __shared__ uint32_t lut[280];
    build_lut4(lut, threadIdx.x);
    __syncthreads();

    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;
    const uint32_t* lutp = lut + (lane & 0x18);   // 4相: 起点 = 8*(lane>>3)

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc = 0.0f;
    __half2 accA = __float2half2_rn(0.0f), accB = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
        uint4 pk = __ldcs(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldcs(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a4 = reinterpret_cast<const uint4*>(activation + (size_t)(b << 4));
        uint4 aa[4] = {__ldg(a4), __ldg(a4 + 1), __ldg(a4 + 2), __ldg(a4 + 3)};

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        __half2 wv[16];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int p = 0; p < 4; p++)
                wv[j * 4 + p] = *reinterpret_cast<const __half2*>(
                    &lutp[(wq[j] >> (8 * p)) & 0xFFu]);

        // 双链交错: j=0,1 -> accA, j=2,3 -> accB
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                __half2 avA = *reinterpret_cast<const __half2*>(&aa[j].x + p);
                __half2 avB = *reinterpret_cast<const __half2*>(&aa[2 + j].x + p);
                accA = __hfma2(wv[j * 4 + p], avA, accA);
                accB = __hfma2(wv[(j + 2) * 4 + p], avB, accB);
            }
        }
        acc += (__low2float(accA) + __high2float(accA)) * e8m0_to_float(sc2 & 0xFFu)
             + (__low2float(accB) + __high2float(accB)) * e8m0_to_float(sc2 >> 8);
        accA = __float2half2_rn(0.0f);
        accB = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) output[warp_global] = __float2half(acc);
}

// ---- GEMV-V7 (T=1, split-K x2): 提升 occupancy ----
// v6 实测: 消 bank conflict 后仍 467 GB/s -> conflict 非真瓶颈。
// 真瓶颈: K=5120 时每 warp 仅 5 迭代, M=31232 只有 976 warp,
//   V100 满载需 5120 warp (80SM x 64) -> occupancy 19%, 延迟无法隐藏。
// 对策: 每行 2 warp 各算半段 K (split-K=2) -> warp x2 (38%), partial 写
//   float 缓冲, 同 stream 串 reduce kernel, 对外接口不变 (M half)。
#ifndef GEMV7_SPLIT
#define GEMV7_SPLIT 2
#endif

__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv_v7_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K]
    float*         __restrict__ partial,         // [SPLIT, M]
    int M, int K)
{
    __shared__ uint32_t lut[280];
    build_lut4(lut, threadIdx.x);
    __syncthreads();

    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M * GEMV7_SPLIT) return;
    const int lane = threadIdx.x & 31;
    const int row = warp_global / GEMV7_SPLIT;
    const int seg = warp_global % GEMV7_SPLIT;
    const uint32_t* lutp = lut + (lane & 0x18);

    const uint8_t* row_pk = weight_packed + (size_t)row * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)row * (size_t)nblocks;

    float acc = 0.0f;
    __half2 accA = __float2half2_rn(0.0f), accB = __float2half2_rn(0.0f);

    const int b_begin = seg * (nblocks / GEMV7_SPLIT);
    const int b_end = (seg + 1) * (nblocks / GEMV7_SPLIT);
    for (int b = b_begin + lane * 2; b < b_end; b += 64) {
        uint4 pk = __ldcs(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldcs(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a4 = reinterpret_cast<const uint4*>(activation + (size_t)(b << 4));
        uint4 aa[4] = {__ldg(a4), __ldg(a4 + 1), __ldg(a4 + 2), __ldg(a4 + 3)};

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        __half2 wv[16];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int p = 0; p < 4; p++)
                wv[j * 4 + p] = *reinterpret_cast<const __half2*>(
                    &lutp[(wq[j] >> (8 * p)) & 0xFFu]);

        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                __half2 avA = *reinterpret_cast<const __half2*>(&aa[j].x + p);
                __half2 avB = *reinterpret_cast<const __half2*>(&aa[2 + j].x + p);
                accA = __hfma2(wv[j * 4 + p], avA, accA);
                accB = __hfma2(wv[(j + 2) * 4 + p], avB, accB);
            }
        }
        acc += (__low2float(accA) + __high2float(accA)) * e8m0_to_float(sc2 & 0xFFu)
             + (__low2float(accB) + __high2float(accB)) * e8m0_to_float(sc2 >> 8);
        accA = __float2half2_rn(0.0f);
        accB = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) partial[(size_t)seg * M + row] = acc;
}

// reduce: out[m] = (half)Σ_s partial[s][m] — 每线程 1 行
__global__ void gemv7_reduce_kernel(
    const float* __restrict__ partial,   // [SPLIT, M]
    __half*      __restrict__ output,    // [M]
    int M)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;
    float s = 0.0f;
    #pragma unroll
    for (int i = 0; i < GEMV7_SPLIT; i++)
        s += partial[(size_t)i * M + m];
    output[m] = __float2half(s);
}

// ---- GEMV-V8 (T=1, PRMT 硬件查表): 完全脱离 LDS ----
// 决定性对照 (p9): v4 随机 LDS 470 vs LUT_BYPASS 广播 837 GB/s
//   -> 16 条随机 LDS 的 bank conflict replay 吃掉 issue slot, 即真瓶颈。
//   4相LUT(v6)无效: 随机索引下固定相位偏移不改变每 bank 负载分布 (Poisson(1) 不变)。
// PRMT 方案: E2M1 幅值 half 高字节表 8 项 {00,38,3C,3E,40,42,44,46}, 低字节恒 0。
//   __byte_perm(T_lo, T_hi, w&0x07070707) 一次查 4 元素 (PRMT 3bit 索引直通);
//   符号 (nibble bit3) 提取为 0x80 字节 OR 进高字节;
//   交织 word_k=[00,lo_k,00,hi_k]: 1 PRMT (sel 常量) + 1 AND 清位。
//   每 32bit 寄存器 (8 fp4 = 4 half2) 共 18 条 ALU = 2.25 条/元素,
//   vs LDS 版 issue 等效 ~4.5/元素, 且无 LDS 依赖延迟。
__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv_v8_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K]
    __half*        __restrict__ output,          // [M]
    int M, int K)
{
    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    // 正幅值高字节表: lo32={00,38,3C,3E} hi32={40,42,44,46} (码点 0-7)
    const uint32_t T_lo = 0x3E3C3800u, T_hi = 0x46444240u;

    float acc = 0.0f;
    __half2 accA = __float2half2_rn(0.0f), accB = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
        uint4 pk = __ldcs(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldcs(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a4 = reinterpret_cast<const uint4*>(activation + (size_t)(b << 4));
        uint4 aa[4] = {__ldg(a4), __ldg(a4 + 1), __ldg(a4 + 2), __ldg(a4 + 3)};

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        __half2 wv[16];
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t w = wq[j];
            // PRMT selector 只用低16位 (4 nibble) — fp4 nibble 天然 nibble 间隔,
            // w & 0x7777 后低16位 = e0-e3 码点直通; 高半段 >>16 查 e4-e7
            uint32_t lo4 = __byte_perm(T_lo, T_hi, w & 0x77777777u);
            uint32_t hi4 = __byte_perm(T_lo, T_hi, (w >> 16) & 0x77777777u);
            // 符号: sl2 byte k bit7 = s_{2k}, sh2 byte k bit7 = s_{2k+1}
            // PRMT 交错成连续符号字节序列
            uint32_t sl2 = (w & 0x08080808u) << 4;
            uint32_t sh2 = w & 0x80808080u;
            lo4 |= __byte_perm(sl2, sh2, 0x5140u);        // [s0,s1,s2,s3]
            hi4 |= __byte_perm(sl2, sh2, 0x7362u);        // [s4,s5,s6,s7]
            // 交织 word_k = [00, e_{2k}, 00, e_{2k+1}]: PRMT 选字节 + AND 清 byte0/2
            uint32_t w0 = __byte_perm(lo4, hi4, 0x1001u) & 0xFF00FF00u;
            uint32_t w1 = __byte_perm(lo4, hi4, 0x3223u) & 0xFF00FF00u;
            uint32_t w2 = __byte_perm(lo4, hi4, 0x5445u) & 0xFF00FF00u;
            uint32_t w3 = __byte_perm(lo4, hi4, 0x7667u) & 0xFF00FF00u;
            wv[j * 4 + 0] = *reinterpret_cast<const __half2*>(&w0);
            wv[j * 4 + 1] = *reinterpret_cast<const __half2*>(&w1);
            wv[j * 4 + 2] = *reinterpret_cast<const __half2*>(&w2);
            wv[j * 4 + 3] = *reinterpret_cast<const __half2*>(&w3);
        }

        // 双链交错: j=0,1 -> accA, j=2,3 -> accB
        #pragma unroll
        for (int j = 0; j < 2; j++) {
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                __half2 avA = *reinterpret_cast<const __half2*>(&aa[j].x + p);
                __half2 avB = *reinterpret_cast<const __half2*>(&aa[2 + j].x + p);
                accA = __hfma2(wv[j * 4 + p], avA, accA);
                accB = __hfma2(wv[(j + 2) * 4 + p], avB, accB);
            }
        }
        acc += (__low2float(accA) + __high2float(accA)) * e8m0_to_float(sc2 & 0xFFu)
             + (__low2float(accB) + __high2float(accB)) * e8m0_to_float(sc2 >> 8);
        accA = __float2half2_rn(0.0f);
        accB = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    if (lane == 0) output[warp_global] = __float2half(acc);
}

// ---- GEMV2 (T=2, MTP verify): 一次权重读出 2 个 token 的输出 ----
// out 布局 (2, M): out[m] = w[m,:]·act0, out[M+m] = w[m,:]·act1
#ifndef GB2_MINB
#define GB2_MINB 8
#endif
__global__ void __launch_bounds__(GEMV_THREADS, GB2_MINB) nvfp4_gemv2_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ act0,            // [K] token A
    const __half*  __restrict__ act1,            // [K] token B
    __half*        __restrict__ output,          // [2, M]
    int M, int K)
{
    __shared__ uint32_t lut[256];
    build_lut(lut, threadIdx.x);
    __syncthreads();

    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc0 = 0.0f, acc1 = 0.0f;
    __half2 a20 = __float2half2_rn(0.0f), a21 = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
        uint4 pk = __ldg(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldg(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a40 = reinterpret_cast<const uint4*>(act0 + (size_t)(b << 4));
        const uint4* a41 = reinterpret_cast<const uint4*>(act1 + (size_t)(b << 4));
        uint4 b0[4] = {__ldg(a40), __ldg(a40 + 1), __ldg(a40 + 2), __ldg(a40 + 3)};
        uint4 b1[4] = {__ldg(a41), __ldg(a41 + 1), __ldg(a41 + 2), __ldg(a41 + 3)};

        // v3.4: 16 次 LDS 前置批量发射 (互不依赖), bank conflict 串行段
        //        与后续 hfma2 重叠 — 对照实验证明 LDS 是首要瓶颈 (53.5->37.5us)
        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        __half2 wv[16];
        #pragma unroll
        for (int j = 0; j < 4; j++)
            #pragma unroll
            for (int p = 0; p < 4; p++)
                wv[j * 4 + p] = *reinterpret_cast<const __half2*>(
                    &LUT_LOOKUP(lut, (wq[j] >> (8 * p)) & 0xFFu));

        #pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t sc = (j < 2) ? (sc2 & 0xFFu) : (sc2 >> 8);
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                __half2 x0 = *reinterpret_cast<const __half2*>(
                    &b0[2 * (j >> 1)].x + (j & 1) * 4 + p);
                __half2 x1 = *reinterpret_cast<const __half2*>(
                    &b1[2 * (j >> 1)].x + (j & 1) * 4 + p);
                a20 = __hfma2(wv[j * 4 + p], x0, a20);
                a21 = __hfma2(wv[j * 4 + p], x1, a21);
            }
            float s = e8m0_to_float(sc);
            acc0 += (__low2float(a20) + __high2float(a20)) * s;
            acc1 += (__low2float(a21) + __high2float(a21)) * s;
            a20 = __float2half2_rn(0.0f);
            a21 = __float2half2_rn(0.0f);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
    }
    if (lane == 0) {
        output[warp_global] = __float2half(acc0);
        output[M + warp_global] = __float2half(acc1);
    }
}

// ---- GEMV2-NL (T=2, v3.5 no-LUT): 位构造反量化, 0 LDS ----
// 对照实验: GEMV2+LUT=276GB/s, GEMV2+LUT_BYPASS=394GB/s -> LDS bank conflict
//   吃掉 30% (T=2 的 hfma2 翻倍使 LDS 延迟更难隐藏, 与 T=1 不同)
// 解法: T=2 弃 LUT, 用发现27 位构造 (~7 INT op/元素, v2 证明≈免费)
__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv2nl_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ act0,            // [K] token A
    const __half*  __restrict__ act1,            // [K] token B
    __half*        __restrict__ output,          // [2, M]
    int M, int K)
{
    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc0 = 0.0f, acc1 = 0.0f;
    __half2 a20 = __float2half2_rn(0.0f), a21 = __float2half2_rn(0.0f);

    for (int b = lane * 2; b < nblocks; b += 64) {
        uint4 pk = __ldg(reinterpret_cast<const uint4*>(row_pk + (size_t)(b << 3)));
        uint32_t sc2 = __ldg(reinterpret_cast<const uint16_t*>(row_sc + b));

        const uint4* a40 = reinterpret_cast<const uint4*>(act0 + (size_t)(b << 4));
        const uint4* a41 = reinterpret_cast<const uint4*>(act1 + (size_t)(b << 4));
        uint4 b0[4] = {__ldg(a40), __ldg(a40 + 1), __ldg(a40 + 2), __ldg(a40 + 3)};
        uint4 b1[4] = {__ldg(a41), __ldg(a41 + 1), __ldg(a41 + 2), __ldg(a41 + 3)};

        uint32_t wq[4] = {pk.x, pk.y, pk.z, pk.w};
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t w = wq[j];
            uint32_t sc = (j < 2) ? (sc2 & 0xFFu) : (sc2 >> 8);
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t byte = (w >> (8 * p)) & 0xFFu;
                // 位构造反量化 (发现27): 低 nibble -> half low, 高 -> half high
                uint32_t n0 = byte & 0xFu, n1 = byte >> 4;
                uint32_t c0 = n0 & 7u, c1 = n1 & 7u;
                uint32_t h0 = (c0 >= 2u) ? (0x3800u + (c0 << 9))
                          : (c0 == 1u) ? 0x3800u : 0u;
                uint32_t h1 = (c1 >= 2u) ? (0x3800u + (c1 << 9))
                          : (c1 == 1u) ? 0x3800u : 0u;
                h0 |= (n0 & 8u) << 12;
                h1 |= (n1 & 8u) << 12;
                uint32_t wpack = h0 | (h1 << 16);
                __half2 wv = *reinterpret_cast<const __half2*>(&wpack);
                __half2 x0 = *reinterpret_cast<const __half2*>(
                    &b0[2 * (j >> 1)].x + (j & 1) * 4 + p);
                __half2 x1 = *reinterpret_cast<const __half2*>(
                    &b1[2 * (j >> 1)].x + (j & 1) * 4 + p);
                a20 = __hfma2(wv, x0, a20);
                a21 = __hfma2(wv, x1, a21);
            }
            float s = e8m0_to_float(sc);
            acc0 += (__low2float(a20) + __high2float(a20)) * s;
            acc1 += (__low2float(a21) + __high2float(a21)) * s;
            a20 = __float2half2_rn(0.0f);
            a21 = __float2half2_rn(0.0f);
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
    }
    if (lane == 0) {
        output[warp_global] = __float2half(acc0);
        output[M + warp_global] = __float2half(acc1);
    }
}

// ---- GEMV2-I (T=2 interleaved, v3.3): 交织 act + dual-LUT broadcast ----
// v3.2 GEMV2 实测 262-286 GB/s (vs GEMV 419-518)。分析:
//   1) b0[4]+b1[4] 双组 act 寄存器 + a20/a21 双累加链 -> 调度压力
//   2) act0/act1 分离加载 -> L1 事务翻倍
// v3.3 改: act 交织布局 [a0[k],a1[k],a0[k+1],a1[k+1],...]
//   + dual-LUT: lutBB[byte]={(h0,h0),(h1,h1)} broadcast 表 (2KB smem)
//   每 byte = 2 次 LDS + 2 次 hfma2 (hfma2 的 .x/.y 同时累积 token0/token1)
//   一次 act 加载服务两个 token, 单条 acc.x=token0 acc.y=token1
//   每 lane 1 block: act 寄存器 8×u32, ~28 regs, 保持 8 blocks/SM
__global__ void __launch_bounds__(GEMV_THREADS, 8) nvfp4_gemv2i_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const uint32_t* __restrict__ act_i,          // [K] 交织 pairs (a0[k],a1[k])
    __half*        __restrict__ output,          // [2, M]
    int M, int K)
{
    __shared__ uint32_t lutBB[512];              // [0..255]=(h0,h0), [256..511]=(h1,h1)
    {
        uint32_t b = threadIdx.x;                // 256 线程, 每 1 byte 建 2 项
        uint32_t n0 = b & 0xFu, n1 = b >> 4;
        uint32_t h0 = e2m1_to_half_bits(n0 & 7u) | ((n0 & 8u) << 12);
        uint32_t h1 = e2m1_to_half_bits(n1 & 7u) | ((n1 & 8u) << 12);
        lutBB[b] = (h0 << 16) | h0;              // (h0, h0)
        lutBB[256 + b] = (h1 << 16) | h1;        // (h1, h1)
    }
    __syncthreads();

    const int nblocks = K >> 4;
    const int warp_global = (blockIdx.x * GEMV_THREADS + threadIdx.x) >> 5;
    if (warp_global >= M) return;
    const int lane = threadIdx.x & 31;

    const uint8_t* row_pk = weight_packed + (size_t)warp_global * (size_t)(K >> 1);
    const uint8_t* row_sc = weight_scales + (size_t)warp_global * (size_t)nblocks;

    float acc0 = 0.0f, acc1 = 0.0f;
    __half2 accA = __float2half2_rn(0.0f), accB = __float2half2_rn(0.0f);

    for (int b = lane; b < nblocks; b += 32) {   // 每 lane 1 block, stride=32
        uint2 pk = __ldg(reinterpret_cast<const uint2*>(row_pk + (size_t)(b << 3)));
        uint32_t sc = __ldg(row_sc + b);

        // 16 权重元素 x 1 pair(=(a0[k],a1[k])) = 16 uint32 = 64B
        const uint32_t* ai = act_i + (size_t)(b << 3) * 2;
        uint32_t av[16];
        #pragma unroll
        for (int p = 0; p < 16; p++) av[p] = __ldg(ai + p);

        const uint32_t wq[2] = {pk.x, pk.y};
        #pragma unroll
        for (int j = 0; j < 2; j++) {            // wq[j] = bytes 4j..4j+3 = 元素 8j..8j+7
            const uint32_t w = wq[j];
            #pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t byte = (w >> (8 * p)) & 0xFFu;
                // 元素 8j+2p -> accA, 元素 8j+2p+1 -> accB (双链 ILP)
                accA = __hfma2(*reinterpret_cast<const __half2*>(&lutBB[byte]),
                               *reinterpret_cast<const __half2*>(&av[8 * j + 2 * p]),
                               accA);
                accB = __hfma2(*reinterpret_cast<const __half2*>(&lutBB[256 + byte]),
                               *reinterpret_cast<const __half2*>(&av[8 * j + 2 * p + 1]),
                               accB);
            }
        }
        // 块尾结算: accA.x/accB.x -> token0, accA.y/accB.y -> token1
        float s = e8m0_to_float(sc);
        acc0 += (__low2float(accA) + __low2float(accB)) * s;
        acc1 += (__high2float(accA) + __high2float(accB)) * s;
        accA = __float2half2_rn(0.0f);
        accB = __float2half2_rn(0.0f);
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc0 += __shfl_down_sync(0xffffffffu, acc0, off);
        acc1 += __shfl_down_sync(0xffffffffu, acc1, off);
    }
    if (lane == 0) {
        output[warp_global] = __float2half(acc0);
        output[M + warp_global] = __float2half(acc1);
    }
}

// ============================================================
// C 接口: 显存管理 + launch (与 v5 DLL 风格一致)
// ============================================================
extern "C" {

__declspec(dllexport) void* gpu_malloc(size_t bytes) {
    void* p = nullptr;
    cudaError_t e = cudaMalloc(&p, bytes);
    if (e != cudaSuccess) { fprintf(stderr, "gpu_malloc: %s\n", cudaGetErrorString(e)); return nullptr; }
    return p;
}

__declspec(dllexport) void gpu_free(void* p) { if (p) cudaFree(p); }

__declspec(dllexport) int gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
}

__declspec(dllexport) int gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    return (int)cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
}

__declspec(dllexport) int launch_nvfp4_gemv(
    const void* weight_packed, const void* weight_scales,
    const void* activation, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;   // K 必须是 32 的倍数
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemv_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)activation, (__half*)output, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;  // 异步发射: 每 token 497 次调用, 由消费端统一同步
}

// GEMV-V5: global-LUT 版 T=1 — DLL 内自管 1KB 表 (首次调用 lazy 初始化)
__declspec(dllexport) int launch_nvfp4_gemv_v5(
    const void* weight_packed, const void* weight_scales,
    const void* activation, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;
    static uint32_t* glut = nullptr;
    if (!glut) {
        uint32_t h[256];
        for (uint32_t b = 0; b < 256; b++) {
            uint32_t n0 = b & 0xFu, n1 = b >> 4;
            uint32_t h0 = e2m1_to_half_bits(n0 & 7u) | ((n0 & 8u) << 12);
            uint32_t h1 = e2m1_to_half_bits(n1 & 7u) | ((n1 & 8u) << 12);
            h[b] = (h1 << 16) | h0;
        }
        if (cudaMalloc(&glut, 1024) != cudaSuccess) return -3;
        cudaMemcpy(glut, h, 1024, cudaMemcpyHostToDevice);
    }
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemv_v5_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)activation, (__half*)output, (const uint32_t*)glut, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv_v5 launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

// GEMV-V6: 4相 LUT 消 conflict 版 T=1 — D1 实验入口
__declspec(dllexport) int launch_nvfp4_gemv_v6(
    const void* weight_packed, const void* weight_scales,
    const void* activation, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemv_v6_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)activation, (__half*)output, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv_v6 launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

// GEMV-V7: split-K x2 版 T=1 — 内部 partial buffer + reduce kernel 串接
__declspec(dllexport) int launch_nvfp4_gemv_v7(
    const void* weight_packed, const void* weight_scales,
    const void* activation, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2 * GEMV7_SPLIT) != 0) return -2;
    static float* partial = nullptr;
    static int cap = 0;
    if (cap < M * GEMV7_SPLIT) {
        if (partial) cudaFree(partial);
        if (cudaMalloc(&partial, sizeof(float) * M * GEMV7_SPLIT) != cudaSuccess) return -3;
        cap = M * GEMV7_SPLIT;
    }
    dim3 block(GEMV_THREADS);
    int total_warps = M * GEMV7_SPLIT;
    dim3 grid((unsigned)((total_warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemv_v7_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)activation, partial, M, K);
    int rthreads = 256;
    dim3 rgrid((unsigned)((M + rthreads - 1) / rthreads));
    gemv7_reduce_kernel<<<rgrid, rthreads, 0, (cudaStream_t)stream>>>(
        partial, (__half*)output, M);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv_v7 launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射 (两 kernel 同 stream 保序)
}

// GEMV-V8: PRMT 硬件查表版 T=1 — 无 LDS 无冲突
__declspec(dllexport) int launch_nvfp4_gemv_v8(
    const void* weight_packed, const void* weight_scales,
    const void* activation, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemv_v8_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)activation, (__half*)output, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv_v8 launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

// GEMV-V4: D1 实验入口 — LDS 隐藏 + __ldcs 分流版 T=1
__declspec(dllexport) int launch_nvfp4_gemv_v4(
    const void* weight_packed, const void* weight_scales,
    const void* activation, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemv_v4_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)activation, (__half*)output, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv_v4 launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

// GEMV-NL (v4): D1 实验入口 — 位构造版 T=1 (与 launch_nvfp4_gemv 接口一致)
__declspec(dllexport) int launch_nvfp4_gemvnl(
    const void* weight_packed, const void* weight_scales,
    const void* activation, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemvnl_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)activation, (__half*)output, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemvnl launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

__declspec(dllexport) int launch_nvfp4_gemv2(
    const void* weight_packed, const void* weight_scales,
    const void* act0, const void* act1, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    // v3.5: GEMV2-NL (位构造, 0 LDS) — 实测 276->? GB/s, 见 NVFP4_研究记忆 发现41
    nvfp4_gemv2nl_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const __half*)act0, (const __half*)act1, (__half*)output, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv2 launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

// GEMV2-I: act_i 为交织布局 [a0[0],a1[0],a0[1],a1[1],...] (K pairs, uint32 视角)
__declspec(dllexport) int launch_nvfp4_gemv2i(
    const void* weight_packed, const void* weight_scales,
    const void* act_i, void* output,
    int M, int K, void* stream)
{
    if (K % (NVFP4_BLOCK_SIZE * 2) != 0) return -2;
    dim3 block(GEMV_THREADS);
    dim3 grid((unsigned)((M + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK));
    nvfp4_gemv2i_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        (const uint8_t*)weight_packed, (const uint8_t*)weight_scales,
        (const uint32_t*)act_i, (__half*)output, M, K);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "gemv2i launch: %s\n", cudaGetErrorString(e)); return -1; }
    return 0;   // 异步发射
}

}   // extern "C"
