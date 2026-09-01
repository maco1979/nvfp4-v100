/*
 * NVFP4 Fused Dequant-GEMM Kernel v5 — 阶段4 优化第3档: TensorCore wmma
 * ============================================================
 * 相对 v4 的改动 (对照实验, v4 保留):
 *   1. 计算路径换 TensorCore: nvcuda::wmma 16x16x16 (sm_70 无 ldmatrix,
 *      wmma API 自动处理 fragment 布局, 编译为 mma.m16n8k16 指令对)
 *      - v4: CUDA core __hfma2, 峰值 31.4TF (实测 64%)
 *      - v5: V100 TensorCore FP16, 峰值 125TF
 *   2. tile 放大: TILE_K 32->64, 每 k-tile 计算量 x2, sync 开销减半
 *      - 8 warps (2x4 网格), warp tile 32x16 = 2 wmma 累加器 fragment
 *      - 每 k-tile 每 warp: 4 k-step x 2 = 8 次 mma_sync
 *   3. TC 内建 FP32 累加器 (fragment<accumulator,...,float>):
 *      - 精度回到 v3 纯 float 路径水平 (优于 v4 两级累加, 发现24)
 *   4. W/A 行 padding (W 40 half=80B, A 136 half=272B, 均 16B 对齐且 ldm 8 倍数):
 *      消 wmma load_matrix_sync 的 bank conflict (行距 words 与 32 bank 错开)
 *   5. 三级软件流水线 (v5.1, 发现26): global LDG 先入寄存器暂存,
 *      TC compute 之后才反量化写 shared
 *   6. v5.2 (发现27): E2M1 反量化位构造 + __hmul2, 替代 constant 查表
 *      (constant cache warp 内地址发散串行化), 1.07x -> 1.17x v4
 *   7. v5.3: TILE_N 64->128, TILE_K 64->32, warp tile 32x16->32x32
 *      (fragment 复用翻倍: LDS/HMMA 0.75->0.50, LDS 压力 -33%);
 *      epilogue 改 half 借 shared 存 (64x128 float=32KB 放不下 -> half 16KB)
 *
 * shared 用量 (v5.3): W_buf[2][64][40] + A_buf[2][32][136] = 10240+17408 = 27KB
 *   < 48KB 静态上限; V100 96KB/SM -> 3 block/SM (若寄存器允许)
 *
 * 编译: nvcc -shared -o nvfp4_cuda_v5.dll nvfp4_cuda_v5.cu -O3 -arch=sm_70
 * 验证: nvfp4_cuda_v5_test.py (真值 + v4/v5 一致性 + v4/v5/FP16 三方性能)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdint>

using namespace nvcuda;

#define NVFP4_BLOCK_SIZE 16
#define E8M0_BIAS 127

// v5.2: E2M1 -> half 纯位构造 (替代 v5.1 的 constant 查表)
//   code>=2: half = 0x3800 + (code<<9)  (2.0->0x4000, 6.0->0x4600 逐点验证)
//   code==1: subnormal 0.5 -> 0x3800
//   code==0: 零 -> 0
//   sign bit3 -> half bit15
// 动机 (发现27): constant 查表 E2M1_CODEPOINTS[nib&7] 在 warp 内 32 线程地址
//   发散, constant cache 每周期仅广播 1 地址 -> 串行化; 加 float 乘 scale +
//   __float2half_rn, 每元素 ~10 指令。位构造 + __hmul2 打包降到 ~3 指令/元素。
__device__ __forceinline__ uint32_t e2m1_to_half_bits(uint32_t code) {
    return (code >= 2u) ? (0x3800u + (code << 9))
         : (code == 1u) ? 0x3800u : 0u;
}

// v5 tile 配置 (v5.3: TILE_N 64->128, TILE_K 64->32, warp tile 32x32)
#define TILE_M 64
#define TILE_N 128
#define TILE_K 32
// P2-1: NUM_STAGES 可由命令行 -D 覆盖 (2=双缓冲基线, 3=环形前瞻2tile)
//   3-stage: shared 41472B (<48KB 静态上限), regs +9/thread (~105<128, 不降
//   occupancy 2blocks/SM); LDG 提前 2 tile 发起, 飞行窗口 = 1 个 compute tile
#ifndef NUM_STAGES
#define NUM_STAGES 2
#endif
#define W_STRIDE (TILE_K + 8)    // 40 half = 80B = 5x16B (ldm 8 倍数 + 消 bank conflict)
#define A_STRIDE (TILE_N + 8)    // 136 half = 272B = 17x16B (ldm 8 倍数 + 消 bank conflict)

// warp 划分: 2x4 = 8 warps, warp tile 32x32 (v5.3: fragment 复用翻倍)
//   LDS/HMMA: v5.2 warp 32x16 = 3 loads/4 HMMA (0.75)
//             v5.3 warp 32x32 = 4 loads/8 HMMA (0.50) -> LDS 压力 -33%
#define WARPS_M 2
#define WARPS_N 4
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 16
#define WARP_TILE_M (TILE_M / WARPS_M)          // 32
#define WARP_TILE_N (TILE_N / WARPS_N)          // 32
#define MMA_TILES_M (WARP_TILE_M / WMMA_M)      // 2
#define MMA_TILES_N (WARP_TILE_N / WMMA_N)      // 2

// ============================================================
// 三级流水线: 阶段1 — global 加载到寄存器暂存 (发出后不依赖使用)
// ============================================================
// 线程分工 (v5.3, 256 线程, tile 64x128x32):
//   权重 64m x 32k -> 每线程 8 nibble = 1 uint32 (mw = tid/4, kw = (tid%4)*8)
//   激活 32k x 128n -> 每线程 16 half = 2 个 float4 (ka = tid/8, na = (tid%8)*16)
struct StageRegs {
    uint32_t pk;        // 1 chunk x 8 nibble (TILE_K=32: 每线程 1 chunk)
    uint32_t sc;        // 1 个 scale byte (kw 为 8 倍数, 8 元素 chunk 在 16 block 内)
    float4   a[2];      // 16 half 激活
};

__device__ __forceinline__ void ldg_stage_v5(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K, N]
    int M, int N, int K,
    int block_m, int block_n, int k_base,
    int mw, int kw, int ka, int na,
    StageRegs& r)
{
    // ---- 权重: 1 chunk 原始 nibble (不反量化, 只取数) ----
    int gm = block_m + mw;
    if (gm < M) {
        int gk = k_base + kw;   // 8 对齐
        uint32_t pk = 0;
        if (gk + 8 <= K) {
            pk = *reinterpret_cast<const uint32_t*>(
                weight_packed + (size_t)gm * (K / 2) + gk / 2);
        } else {
            for (int i = 0; i < 8; i++) {
                int kk = gk + i;
                if (kk < K) {
                    uint8_t b = weight_packed[(size_t)gm * (K / 2) + kk / 2];
                    uint8_t nib = (kk % 2 == 0) ? (b & 0x0F) : (b >> 4);
                    pk |= (uint32_t)nib << (4 * i);
                }
            }
        }
        r.pk = pk;
        r.sc = (uint32_t)((gk < K)
            ? weight_scales[(size_t)gm * (K / NVFP4_BLOCK_SIZE)
                            + gk / NVFP4_BLOCK_SIZE] : 127);
    } else {
        r.pk = 0;
        r.sc = 127;   // scale=1.0, 值=0
    }

    // ---- 激活: 16 half 原始直取 ----
    int gk = k_base + ka;
    int gn = block_n + na;
    if (gk < K) {
        if (N % 8 == 0 && gn + 16 <= N) {
            r.a[0] = *reinterpret_cast<const float4*>(
                activation + (size_t)gk * N + gn);
            r.a[1] = *reinterpret_cast<const float4*>(
                activation + (size_t)gk * N + gn + 8);
        } else {
            __half tmp[16];
            #pragma unroll
            for (int i = 0; i < 16; i++)
                tmp[i] = (gn + i < N)
                    ? activation[(size_t)gk * N + gn + i] : __float2half(0.0f);
            r.a[0] = *reinterpret_cast<const float4*>(&tmp[0]);
            r.a[1] = *reinterpret_cast<const float4*>(&tmp[8]);
        }
    } else {
        r.a[0] = make_float4(0, 0, 0, 0);
        r.a[1] = make_float4(0, 0, 0, 0);
    }
}

// ============================================================
// 三级流水线: 阶段3 — 寄存器反量化 + 写 shared (compute 之后执行)
// ============================================================
__device__ __forceinline__ void dequant_store_v5(
    __half (*__restrict__ W_tile)[W_STRIDE],
    __half (*__restrict__ A_tile)[A_STRIDE],
    int mw, int kw, int ka, int na,
    const StageRegs& r)
{
    // ---- 权重: 1 chunk (8 nibble) 位构造反量化 (v5.2 发现27) ----
    {
        __half2 s2 = __half2half2(
            __float2half_rn(exp2f((float)r.sc - (float)E8M0_BIAS)));
        __half2 v[4];
        #pragma unroll
        for (int p = 0; p < 4; p++) {
            uint32_t pair = (r.pk >> (8 * p)) & 0xFF;
            uint32_t n0 = pair & 0xF, n1 = (pair >> 4) & 0xF;
            uint32_t h0 = e2m1_to_half_bits(n0 & 7u) | ((n0 & 8u) << 12);
            uint32_t h1 = e2m1_to_half_bits(n1 & 7u) | ((n1 & 8u) << 12);
            __half2 hv = __halves2half2(__ushort_as_half((unsigned short)h0),
                                        __ushort_as_half((unsigned short)h1));
            v[p] = __hmul2(hv, s2);
        }
        *reinterpret_cast<float4*>(&W_tile[mw][kw]) =
            *reinterpret_cast<const float4*>(&v[0]);
    }
    *reinterpret_cast<float4*>(&A_tile[ka][na]) = r.a[0];
    *reinterpret_cast<float4*>(&A_tile[ka][na + 8]) = r.a[1];
}

// ============================================================
// Kernel: NVFP4 Fused Dequant-GEMM v5 (TensorCore wmma + 三级流水线)
// ============================================================
// v5.3.1 试验: launch_bounds(256,3) 强制 80regs+40B spill -> 1.48x (更慢,
//   spill 代价 > occupancy 收益), 回退 (256) = 96 regs 2 block/SM = 1.72x
__global__ void __launch_bounds__(256) nvfp4_gemmtc_v5_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2] nibble=[sign|code]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16] E8M0
    const __half*  __restrict__ activation,      // [K, N] FP16
    __half*        __restrict__ output,          // [M, N] FP16
    int M, int N, int K)
{
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    // 加载阶段坐标 (v5.3: TILE_K=32, TILE_N=128)
    const int mw = tid / 4;          // 权重行 (0..63)
    const int kw = (tid % 4) * 8;    // 权重 k 段基 (0/8/16/24)
    const int ka = tid / 8;          // 激活 k 行 (0..31)
    const int na = (tid % 8) * 16;   // 激活 n 段基 (0/16/.../112)
    // 计算阶段 warp 坐标 (warp tile 32x32)
    const int warp_id = tid / 32;
    const int warp_row = warp_id / WARPS_N;   // 0..1
    const int warp_col = warp_id % WARPS_N;   // 0..3
    const int wm = warp_row * WARP_TILE_M;    // 0 / 32
    const int wn = warp_col * WARP_TILE_N;    // 0/32/64/96

    __shared__ __align__(16) __half W_buf[NUM_STAGES][TILE_M][W_STRIDE];
    __shared__ __align__(16) __half A_buf[NUM_STAGES][TILE_K][A_STRIDE];

    // wmma 累加器 (TensorCore 内建 FP32 累加, 跨全部 k-tile 保持)
    // v5.3: 2x2 fragments = 32 float regs/warp-thread
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        acc[MMA_TILES_M][MMA_TILES_N];
    #pragma unroll
    for (int i = 0; i < MMA_TILES_M; i++)
        #pragma unroll
        for (int j = 0; j < MMA_TILES_N; j++)
            #pragma unroll
            for (int t = 0; t < acc[i][j].num_elements; t++)
                acc[i][j].x[t] = 0.0f;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;

#if NUM_STAGES == 3
    // ============================================================
    // P2-1 环形 3-stage (V100 无 cp.async, LDG->regs->dequant->smem):
    //   迭代 k: [ldg t_{k+2}->rc] [compute buf k%3] [dequant rb->buf (k+1)%3]
    //   LDG 飞行窗口从 "本 tile compute" 拉长到 "整个下一 tile compute",
    //   语义: 数值路径与 2-stage 完全相同 -> 输出应 bit-exact
    //   buffer 冲突检查: 读 k%3, 写 (k+1)%3, 下轮写 (k+2)%3, 三者互异
    // ============================================================
    StageRegs regs_a, regs_b, regs_c;
    if (num_k_tiles > 0) {
        ldg_stage_v5(weight_packed, weight_scales, activation,
                     M, N, K, block_m, block_n, 0, mw, kw, ka, na, regs_a);
        dequant_store_v5(W_buf[0], A_buf[0], mw, kw, ka, na, regs_a);
        __syncthreads();
    }
    if (num_k_tiles > 1) {
        ldg_stage_v5(weight_packed, weight_scales, activation,
                     M, N, K, block_m, block_n, TILE_K, mw, kw, ka, na, regs_b);
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const int rd = k_tile % 3;
        const int wr = (k_tile + 1) % 3;
        const bool has_next  = (k_tile + 1 < num_k_tiles);
        const bool has_next2 = (k_tile + 2 < num_k_tiles);

        // 阶段1: 提前 2 tile 发起 global 加载 (延迟窗口 = compute(tile k+1) 全程)
        if (has_next2) {
            ldg_stage_v5(weight_packed, weight_scales, activation,
                         M, N, K, block_m, block_n, (k_tile + 2) * TILE_K,
                         mw, kw, ka, na, regs_c);
        }

        // 阶段2: TensorCore 计算 (同 2-stage: 每 k-step 4 次 mma_sync)
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                           wmma::row_major> fa[MMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                           wmma::row_major> fb[MMA_TILES_N];
            #pragma unroll
            for (int i = 0; i < MMA_TILES_M; i++)
                wmma::load_matrix_sync(fa[i], &W_buf[rd][wm + i * WMMA_M][kk],
                                       W_STRIDE);
            #pragma unroll
            for (int j = 0; j < MMA_TILES_N; j++)
                wmma::load_matrix_sync(fb[j], &A_buf[rd][kk][wn + j * WMMA_N],
                                       A_STRIDE);
            #pragma unroll
            for (int i = 0; i < MMA_TILES_M; i++)
                #pragma unroll
                for (int j = 0; j < MMA_TILES_N; j++)
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }

        // 阶段3: 上轮取回数据反量化写 wr buffer, 寄存器组轮转
        if (has_next) {
            dequant_store_v5(W_buf[wr], A_buf[wr], mw, kw, ka, na, regs_b);
            regs_b = regs_c;
        }

        __syncthreads();
    }
#else
    int rd = 0, wr = 1;

    // 三级流水线: LDG(tile0) -> dequant_store -> sync
    StageRegs regs, regs_next;
    if (num_k_tiles > 0) {
        ldg_stage_v5(weight_packed, weight_scales, activation,
                     M, N, K, block_m, block_n, 0, mw, kw, ka, na, regs);
        dequant_store_v5(W_buf[rd], A_buf[rd], mw, kw, ka, na, regs);
        __syncthreads();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        const bool has_next = (k_tile + 1 < num_k_tiles);

        // 阶段1: 发出下一 tile 的 global 加载 (不依赖使用, 延迟被 compute 掩盖)
        if (has_next) {
            ldg_stage_v5(weight_packed, weight_scales, activation,
                         M, N, K, block_m, block_n, (k_tile + 1) * TILE_K,
                         mw, kw, ka, na, regs_next);
        }

        // 阶段2: TensorCore 计算 (v5.3: 每 k-step 4 次 mma_sync, fragment 复用)
        #pragma unroll
        for (int kk = 0; kk < TILE_K; kk += WMMA_K) {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half,
                           wmma::row_major> fa[MMA_TILES_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half,
                           wmma::row_major> fb[MMA_TILES_N];
            #pragma unroll
            for (int i = 0; i < MMA_TILES_M; i++)
                wmma::load_matrix_sync(fa[i], &W_buf[rd][wm + i * WMMA_M][kk],
                                       W_STRIDE);
            #pragma unroll
            for (int j = 0; j < MMA_TILES_N; j++)
                wmma::load_matrix_sync(fb[j], &A_buf[rd][kk][wn + j * WMMA_N],
                                       A_STRIDE);
            #pragma unroll
            for (int i = 0; i < MMA_TILES_M; i++)
                #pragma unroll
                for (int j = 0; j < MMA_TILES_N; j++)
                    wmma::mma_sync(acc[i][j], fa[i], fb[j], acc[i][j]);
        }

        // 阶段3: 上一阶段取回的数据反量化写 wr buffer (此时 TC 已算完 rd)
        if (has_next) {
            dequant_store_v5(W_buf[wr], A_buf[wr], mw, kw, ka, na, regs_next);
        }

        __syncthreads();
        rd ^= 1;
        wr ^= 1;
    }
#endif

    // epilogue (v5.3): C tile 64x128 float = 32KB 放不下 27.6KB shared
    //   -> 按 warp_row 分两轮, 每轮 32 行 x (TILE_N+8) float = 17.4KB 借 W_buf 区域
    //   (store_matrix_sync 要求指针类型与 fragment 元素一致: float acc 只能存 float*)
    const int CFS = TILE_N + 8;   // 136 float 行距 (16B 对齐: 544B = 34x16)
    #pragma unroll
    for (int r = 0; r < WARPS_M; r++) {
        __syncthreads();
        // 借 A_buf 全区域: 2x32x136 half = 17408B = 32 行 x 136 float 严格相等
        // (W_buf 只有 10.2KB 放不下; 主循环已结束, A_buf 复用安全)
        float* Cf = reinterpret_cast<float*>(&A_buf[0][0][0]);
        if (warp_row == r) {
            #pragma unroll
            for (int i = 0; i < MMA_TILES_M; i++)
                #pragma unroll
                for (int j = 0; j < MMA_TILES_N; j++)
                    wmma::store_matrix_sync(
                        &Cf[(size_t)i * WMMA_M * CFS + wn + j * WMMA_N],
                        acc[i][j], CFS, wmma::mem_row_major);
        }
        __syncthreads();

        // guarded 写回: 每轮 32 行 x 128 列, 每线程 16 元素
        // 4 float -> 2x __floats2half2_rn -> uint32 写 (float 位模式不能直写 half!)
        const int row = tid / 8;                                   // 轮内行 0..31
        const int gm_out = block_m + r * WARP_TILE_M + row;
        const int cn = (tid % 8) * 16;                             // 列 0..112
        #pragma unroll
        for (int seg = 0; seg < 16; seg += 4) {
            int gn0 = block_n + cn + seg;
            const float2 f01 = *reinterpret_cast<const float2*>(
                &Cf[(size_t)row * CFS + cn + seg]);
            const float2 f23 = *reinterpret_cast<const float2*>(
                &Cf[(size_t)row * CFS + cn + seg + 2]);
            // 每个uint32=2half: h01/h23 各写一次 (4 float -> 4 half, 2 次 uint32)
            // 对齐: half index 偶 -> 4B 对齐 (N 偶且 gn0 偶)
            if (gm_out < M && gn0 + 4 <= N && (N & 1) == 0) {
                __half2 h01 = __floats2half2_rn(f01.x, f01.y);
                __half2 h23 = __floats2half2_rn(f23.x, f23.y);
                uint32_t* dst = reinterpret_cast<uint32_t*>(
                    &output[(size_t)gm_out * N + gn0]);
                dst[0] = reinterpret_cast<const uint32_t&>(h01);  // 列 +0,+1
                dst[1] = reinterpret_cast<const uint32_t&>(h23);  // 列 +2,+3
            } else {
                float f[4] = {f01.x, f01.y, f23.x, f23.y};
                #pragma unroll
                for (int c = 0; c < 4; c++) {
                    int gn = gn0 + c;
                    if (gm_out < M && gn < N)
                        output[(size_t)gm_out * N + gn] = __float2half_rn(f[c]);
                }
            }
        }
    }
}

// ============================================================
// Host 函数 (extern "C" 导出, ctypes 可调)
// ============================================================
extern "C" {

__declspec(dllexport) int launch_nvfp4_gemmtc_v5(
    const uint8_t* d_weight_packed,
    const uint8_t* d_weight_scales,
    const __half*  d_activation,
    __half*        d_output,
    int M, int N, int K, void* stream)
{
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(256);

    nvfp4_gemmtc_v5_kernel<<<grid, block, 0, (cudaStream_t)stream>>>(
        d_weight_packed, d_weight_scales, d_activation, d_output, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    return 0;   // 异步发射, 消费端同步 (prefill 497 次调用逐次同步开销大)
}

__declspec(dllexport) void* gpu_malloc(size_t bytes) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return nullptr;
    return ptr;
}

__declspec(dllexport) void gpu_free(void* ptr) { cudaFree(ptr); }

// P2-1: bench 用 — launch 异步发射, 需显式同步才能测到真实 kernel 执行时间
__declspec(dllexport) int gpu_sync() {
    return (cudaDeviceSynchronize() == cudaSuccess) ? 0 : (int)cudaErrorUnknown;
}

__declspec(dllexport) int gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    return (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess)
           ? 0 : (int)cudaErrorUnknown;
}

__declspec(dllexport) int gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    return (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess)
           ? 0 : (int)cudaErrorUnknown;
}

__declspec(dllexport) int get_gpu_info(char* name_buf, int buf_size,
                                       int* cc_major, int* cc_minor) {
    int n;
    if (cudaGetDeviceCount(&n) != cudaSuccess || n == 0) return -1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -2;
    int i = 0;
    while (prop.name[i] != '\0' && i < buf_size - 1) { name_buf[i] = prop.name[i]; i++; }
    name_buf[i] = '\0';
    *cc_major = prop.major;
    *cc_minor = prop.minor;
    return 0;
}

// v5 shared 占用: W_buf 18432B + A_buf 18432B = 36864B (对比 v4 的 18KB)
__declspec(dllexport) int get_kernel_shared_mem_bytes() {
    return NUM_STAGES * TILE_M * W_STRIDE * 2
         + NUM_STAGES * TILE_K * A_STRIDE * 2;
}

}  // extern "C"
