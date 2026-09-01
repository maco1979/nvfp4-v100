/*
 * NVFP4 Fused Dequant-GEMM Kernel v4 — 阶段4 优化第2档: __hfma2 half2 打包计算
 * ============================================================
 * 相对 v3 的改动 (对照实验, v3 保留):
 *   1. 计算路径换 __hfma2: 数据保持 half2 打包, 2 FMA/指令, 免 half->float 转换
 *      - v3: 每 lk = 8 ld.shared.u16 + 8 cvt.f32.f16 + 16 FFMA (FP32 路径 15.7TF 峰值)
 *      - v4: 每 lk = 4 ld.shared.u16 + 2 ld.shared.u32 + 8 HFMA2 (FP16 路径 31.4TF 峰值)
 *      - 发现22: v3 实测 11.6TF 已达 FP32 路径 74%, 必须换路径
 *   2. W_buf 行 padding +8 half (行距 40 half = 80B, 仍 16B 对齐):
 *      - 消除计算阶段 W 标量读 4-way bank conflict (无 padding: bank=(16m+lk/2)%32,
 *        m=0..7 两两同 bank; 有 padding: bank=(20m+lk/2)%32, m=0..7 全错开)
 *   3. 两级累加保精度: half2 累加器跑满一个 k-tile (32 步) 后 flush 到
 *      float32 外层累加器 -> K=188160 长累加的 half 舍入误差有界 (不跨 tile 传播)
 *   4. 加载路径/双缓冲/epilogue 与 v3 完全一致
 *
 * shared 用量: W_buf[2][64][40] + A_buf[2][32][64] = 10240+8192 = 18432 B/block (~18KB)
 *   V100 96KB/SM -> 5 block/SM = 1280 线程/SM (v3: 6 block, 计算受限下影响可忽略)
 *
 * 编译: nvcc -shared -o nvfp4_cuda_v4.dll nvfp4_cuda_v4.cu -O3 -arch=sm_70
 * 验证: nvfp4_cuda_v4_test.py (真值 vs Python 参考 + v3/v4 一致性 + 性能对比)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define NVFP4_BLOCK_SIZE 16
#define E8M0_BIAS 127

// E2M1 码点绝对值表 (8 个码点, 索引 0-7)
__constant__ float E2M1_CODEPOINTS[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// v4 tile 配置 (与 v3 相同)
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define THREAD_ROWS 16   // 线程网格 16×16 = 256 线程
#define THREAD_COLS 16
#define REG_M 4          // 每线程输出行数 (64/16)
#define REG_N 4          // 每线程输出列数 (64/16)
#define NUM_STAGES 2

// W 行 padding: 消 bank conflict (见文件头注释 2)
#define W_STRIDE (TILE_K + 8)   // 40 half = 80B, 16B 对齐 (80 = 5×16)

// ============================================================
// 加载一个 k-tile: 反量化权重 + 直传激活 (向量化, 与 v3 相同)
// ============================================================
__device__ __forceinline__ void load_tiles_v4(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K, N]
    __half (*__restrict__ W_tile)[W_STRIDE],     // 单 stage: [TILE_M][W_STRIDE]
    __half (*__restrict__ A_tile)[TILE_N],       // 单 stage: [TILE_K][TILE_N]
    int M, int N, int K,
    int block_m, int block_n, int k_base,
    int mw, int kw, int ka, int na)
{
    // ---- 权重: 反量化 8 个连续 FP4 -> FP16, 写 W_tile[mw][kw..kw+7] ----
    int gm = block_m + mw;
    if (gm < M) {
        int gk = k_base + kw;   // kw 8对齐, 8元素段必在同一 16-block 内
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
        uint8_t scale_byte = (gk < K)
            ? weight_scales[(size_t)gm * (K / NVFP4_BLOCK_SIZE) + gk / NVFP4_BLOCK_SIZE]
            : 127;
        float scale = exp2f((float)scale_byte - (float)E8M0_BIAS);

        __half tmp[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t nib = (pk >> (4 * i)) & 0x0F;
            float abs_val = E2M1_CODEPOINTS[nib & 0x07];
            float val = (nib & 0x08) ? -abs_val : abs_val;
            tmp[i] = __float2half_rn(val * scale);
        }
        // 16B 对齐向量化写: 行基址 80B×mw (16B 倍数) + kw×2B (16B 倍数)
        *reinterpret_cast<float4*>(&W_tile[mw][kw]) =
            *reinterpret_cast<const float4*>(&tmp[0]);
    } else {
        *reinterpret_cast<float4*>(&W_tile[mw][kw]) = make_float4(0, 0, 0, 0);
    }

    // ---- 激活: FP16 直传 8 个 -> A_tile[ka][na..na+7] ----
    int gk = k_base + ka;
    int gn = block_n + na;
    if (gk < K) {
        if (N % 8 == 0 && gn + 8 <= N) {
            *reinterpret_cast<float4*>(&A_tile[ka][na]) =
                *reinterpret_cast<const float4*>(activation + (size_t)gk * N + gn);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                A_tile[ka][na + i] = (gn + i < N)
                    ? activation[(size_t)gk * N + gn + i] : __float2half(0.0f);
        }
    } else {
        *reinterpret_cast<float4*>(&A_tile[ka][na]) = make_float4(0, 0, 0, 0);
    }
}

// ============================================================
// Kernel: NVFP4 Fused Dequant-GEMM v4 (__hfma2 half2 打包计算)
// ============================================================
__global__ void __launch_bounds__(256) nvfp4_gemmtc_v4_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2] nibble=[sign|code]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16] E8M0
    const __half*  __restrict__ activation,      // [K, N] FP16
    __half*        __restrict__ output,          // [M, N] FP16
    int M, int N, int K)
{
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    const int tr = tid / THREAD_COLS;
    const int tc = tid % THREAD_COLS;
    const int m0 = tr * REG_M;
    const int n0 = tc * REG_N;
    const int mw = tid / 4;
    const int kw = (tid % 4) * 8;
    const int ka = tid / 8;
    const int na = (tid % 8) * 8;

    __shared__ __align__(16) __half W_buf[NUM_STAGES][TILE_M][W_STRIDE];
    __shared__ __align__(16) __half A_buf[NUM_STAGES][TILE_K][TILE_N];

    // 两级累加器: half2 内层 (单 k-tile) + float32 外层 (全程)
    float accum[REG_M][REG_N];
    __half2 acc2[REG_M][REG_N / 2];
    #pragma unroll
    for (int i = 0; i < REG_M; i++)
        #pragma unroll
        for (int j = 0; j < REG_N; j++) accum[i][j] = 0.0f;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    int rd = 0, wr = 1;

    if (num_k_tiles > 0) {
        load_tiles_v4(weight_packed, weight_scales, activation,
                      W_buf[rd], A_buf[rd], M, N, K,
                      block_m, block_n, 0, mw, kw, ka, na);
        __syncthreads();
    }

    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        if (k_tile + 1 < num_k_tiles) {
            load_tiles_v4(weight_packed, weight_scales, activation,
                          W_buf[wr], A_buf[wr], M, N, K,
                          block_m, block_n, (k_tile + 1) * TILE_K,
                          mw, kw, ka, na);
        }

        // 内层 half2 累加器清零 (上一 tile 已 flush)
        #pragma unroll
        for (int i = 0; i < REG_M; i++)
            #pragma unroll
            for (int p = 0; p < REG_N / 2; p++)
                acc2[i][p] = __float2half2_rn(0.0f);

        // 计算: 每 lk 一次 4W 标量读 + 2A half2 读 + 8 __hfma2 (16 FMA)
        #pragma unroll
        for (int lk = 0; lk < TILE_K; lk++) {
            __half w[REG_M];
            __half2 a2[REG_N / 2];
            #pragma unroll
            for (int i = 0; i < REG_M; i++)
                w[i] = W_buf[rd][m0 + i][lk];
            #pragma unroll
            for (int p = 0; p < REG_N / 2; p++)
                a2[p] = *reinterpret_cast<const __half2*>(&A_buf[rd][lk][n0 + 2 * p]);
            #pragma unroll
            for (int i = 0; i < REG_M; i++) {
                __half2 w2 = __half2half2(w[i]);
                #pragma unroll
                for (int p = 0; p < REG_N / 2; p++)
                    acc2[i][p] = __hfma2(w2, a2[p], acc2[i][p]);
            }
        }

        // flush: half2 累加器 -> float32 外层 (每 tile 一次, 舍入误差不跨 tile 传播)
        #pragma unroll
        for (int i = 0; i < REG_M; i++)
            #pragma unroll
            for (int p = 0; p < REG_N / 2; p++) {
                float2 f = __half22float2(acc2[i][p]);
                accum[i][2 * p] += f.x;
                accum[i][2 * p + 1] += f.y;
            }

        __syncthreads();
        rd ^= 1;
        wr ^= 1;
    }

    // 写回: 每线程 16 个 FP16 (与 v3 相同)
    const int gm = block_m + m0;
    const int gn = block_n + n0;
    #pragma unroll
    for (int i = 0; i < REG_M; i++) {
        if (gm + i < M) {
            #pragma unroll
            for (int j = 0; j < REG_N; j++) {
                if (gn + j < N)
                    output[(size_t)(gm + i) * N + gn + j] = __float2half_rn(accum[i][j]);
            }
        }
    }
}

// ============================================================
// Host 函数 (extern "C" 导出, ctypes 可调)
// ============================================================
extern "C" {

__declspec(dllexport) int launch_nvfp4_gemmtc_v4(
    const uint8_t* d_weight_packed,
    const uint8_t* d_weight_scales,
    const __half*  d_activation,
    __half*        d_output,
    int M, int N, int K)
{
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(THREAD_ROWS * THREAD_COLS);  // 256 线程

    nvfp4_gemmtc_v4_kernel<<<grid, block>>>(
        d_weight_packed, d_weight_scales, d_activation, d_output, M, N, K);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return (int)err;
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) return (int)err;
    return 0;
}

__declspec(dllexport) void* gpu_malloc(size_t bytes) {
    void* ptr = nullptr;
    if (cudaMalloc(&ptr, bytes) != cudaSuccess) return nullptr;
    return ptr;
}

__declspec(dllexport) void gpu_free(void* ptr) { cudaFree(ptr); }

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

// v4 shared 占用: W_buf 10240B + A_buf 8192B = 18432B (对比 v3 的 16KB)
__declspec(dllexport) int get_kernel_shared_mem_bytes() {
    return NUM_STAGES * TILE_M * W_STRIDE * 2   // W_buf (含 padding)
         + NUM_STAGES * TILE_K * TILE_N * 2;    // A_buf
}

}  // extern "C"
