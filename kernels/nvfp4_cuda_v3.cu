/*
 * NVFP4 Fused Dequant-GEMM Kernel v3 — 阶段4 优化第1档: 寄存器分块 + 向量化加载
 * ============================================================
 * 相对 v2 的改动 (对照实验, v2 保留):
 *   1. 寄存器分块: 每线程 4×4=16 个输出 (v2: 1 个)
 *      - block tile 64×64 (v2: 32×32), 256 线程 (v2: 1024)
 *      - 计算/访存比: 16 FMA / 16B shared 读 = 1.0 (v2: 1 FMA / 4B = 0.25)
 *   2. 向量化 global 加载:
 *      - 权重 packed: uint32 一次读 8 个 FP4 (v2: 逐字节)
 *        (kw 8对齐 => 8元素段永远在同一 16-block 内, 只需 1 个 scale)
 *      - 激活: float4 一次 16B = 8 half (v2: 逐 half)
 *   3. 向量化 shared 写: 反量化结果按 float4×2 写入 (16B 对齐)
 *   4. 双缓冲 + float32 累加器语义与 v2 完全一致 (真值可比)
 *
 * shared 用量: W_buf[2][64][32] + A_buf[2][32][64] = 8+8 = 16 KB/block
 *   V100 96KB/SM -> 6 block/SM = 1536 线程/SM
 *
 * 编译: nvcc -shared -o nvfp4_cuda_v3.dll nvfp4_cuda_v3.cu -O3 -arch=sm_70
 * 验证: nvfp4_cuda_v3_test.py (真值 vs Python 参考 + 性能 v2/v3/FP16 三方对比)
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

// v3 tile 配置
#define TILE_M 64
#define TILE_N 64
#define TILE_K 32
#define THREAD_ROWS 16   // 线程网格 16×16 = 256 线程
#define THREAD_COLS 16
#define REG_M 4          // 每线程输出行数 (64/16)
#define REG_N 4          // 每线程输出列数 (64/16)
#define NUM_STAGES 2

// ============================================================
// 加载一个 k-tile: 反量化权重 + 直传激活 (向量化)
// ============================================================
// 线程分工 (256 线程):
//   权重 64m×32k = 2048 元素 -> 每线程 8 个 (一行内连续 k, tid: m=tid/4, k=(tid%4)*8)
//   激活 32k×64n = 2048 half -> 每线程 8 个 (一行内连续 n, tid: k=tid/8, n=(tid%8)*8)
__device__ __forceinline__ void load_tiles_v3(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16]
    const __half*  __restrict__ activation,      // [K, N]
    __half (*__restrict__ W_tile)[TILE_K],       // 单 stage: [TILE_M][TILE_K]
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
            // gk 8对齐 => gk/2 4对齐; K 为 16 倍数 => gm*(K/2) 4对齐 (安全 uint32 读)
            pk = *reinterpret_cast<const uint32_t*>(
                weight_packed + (size_t)gm * (K / 2) + gk / 2);
        } else {
            // K 边界: 逐 nibble 组装 (越界位保持 0)
            for (int i = 0; i < 8; i++) {
                int kk = gk + i;
                if (kk < K) {
                    uint8_t b = weight_packed[(size_t)gm * (K / 2) + kk / 2];
                    uint8_t nib = (kk % 2 == 0) ? (b & 0x0F) : (b >> 4);
                    pk |= (uint32_t)nib << (4 * i);
                }
            }
        }
        // 该 8 元素段唯一 scale (段在单一 16-block 内; 越界 nibble=0 无影响)
        uint8_t scale_byte = (gk < K)
            ? weight_scales[(size_t)gm * (K / NVFP4_BLOCK_SIZE) + gk / NVFP4_BLOCK_SIZE]
            : 127;
        float scale = exp2f((float)scale_byte - (float)E8M0_BIAS);  // 发现6: byte<=254 安全

        __half tmp[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            uint8_t nib = (pk >> (4 * i)) & 0x0F;
            float abs_val = E2M1_CODEPOINTS[nib & 0x07];
            float val = (nib & 0x08) ? -abs_val : abs_val;   // v2: sign 内嵌 nibble bit3
            tmp[i] = __float2half_rn(val * scale);
        }
        // 16B 对齐向量化写: 8 half = 16B = 恰好一个 float4, 一次写完
        // (kw*2 是 16 的倍数, W_tile 行 stride 64B; 修复: 原两段写第二段 [kw+4,kw+12)
        //  越界 4 half — 发现21, 与 else 分支同源 bug)
        *reinterpret_cast<float4*>(&W_tile[mw][kw]) = *reinterpret_cast<const float4*>(&tmp[0]);
    } else {
        // 填零 8 half = 一次 float4 (kw+8 <= TILE_K 恒成立, 修复: 原循环 i+=4 配 float4
        // 8half 写会越界 [kw+8, kw+12) — 发现21)
        *reinterpret_cast<float4*>(&W_tile[mw][kw]) = make_float4(0, 0, 0, 0);
    }

    // ---- 激活: FP16 直传 8 个 -> A_tile[ka][na..na+7] ----
    int gk = k_base + ka;
    int gn = block_n + na;
    if (gk < K) {
        if (N % 8 == 0 && gn + 8 <= N) {
            // 快路径: N 8倍数 => (gk*N+gn)*2 为 16 的倍数, float4 安全
            *reinterpret_cast<float4*>(&A_tile[ka][na]) =
                *reinterpret_cast<const float4*>(activation + (size_t)gk * N + gn);
        } else {
            #pragma unroll
            for (int i = 0; i < 8; i++)
                A_tile[ka][na + i] = (gn + i < N)
                    ? activation[(size_t)gk * N + gn + i] : __float2half(0.0f);
        }
    } else {
        // 填零 8 half = 一次 float4 (na+8 <= TILE_N 恒成立, 发现21 修复)
        *reinterpret_cast<float4*>(&A_tile[ka][na]) = make_float4(0, 0, 0, 0);
    }
}

// ============================================================
// Kernel: NVFP4 Fused Dequant-GEMM v3 (寄存器分块 + 双缓冲)
// ============================================================
__global__ void __launch_bounds__(256) nvfp4_gemmtc_v3_kernel(
    const uint8_t* __restrict__ weight_packed,   // [M, K/2] nibble=[sign|code]
    const uint8_t* __restrict__ weight_scales,   // [M, K/16] E8M0
    const __half*  __restrict__ activation,      // [K, N] FP16
    __half*        __restrict__ output,          // [M, N] FP16
    int M, int N, int K)
{
    const int block_m = blockIdx.y * TILE_M;
    const int block_n = blockIdx.x * TILE_N;

    const int tid = threadIdx.x;
    // 计算阶段坐标: 线程 (tr, tc) 负责 tile 内 [tr*4..tr*4+3] × [tc*4..tc*4+3]
    const int tr = tid / THREAD_COLS;
    const int tc = tid % THREAD_COLS;
    const int m0 = tr * REG_M;
    const int n0 = tc * REG_N;
    // 加载阶段坐标
    const int mw = tid / 4;        // 权重行 (0..63)
    const int kw = (tid % 4) * 8;  // 权重 k 段基 (0/8/16/24)
    const int ka = tid / 8;        // 激活 k 行 (0..31)
    const int na = (tid % 8) * 8;  // 激活 n 段基 (0/8/.../56)

    __shared__ __align__(16) __half W_buf[NUM_STAGES][TILE_M][TILE_K];
    __shared__ __align__(16) __half A_buf[NUM_STAGES][TILE_K][TILE_N];

    // 寄存器累加器 (float32, 与 v2 语义一致)
    float accum[REG_M][REG_N];
    #pragma unroll
    for (int i = 0; i < REG_M; i++)
        #pragma unroll
        for (int j = 0; j < REG_N; j++) accum[i][j] = 0.0f;

    const int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    int rd = 0, wr = 1;

    // 预加载 k_tile 0
    if (num_k_tiles > 0) {
        load_tiles_v3(weight_packed, weight_scales, activation,
                      W_buf[rd], A_buf[rd], M, N, K,
                      block_m, block_n, 0, mw, kw, ka, na);
        __syncthreads();
    }

    // 主循环: 双缓冲 (load 下一块到 wr, 同时用 rd 计算)
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        if (k_tile + 1 < num_k_tiles) {
            load_tiles_v3(weight_packed, weight_scales, activation,
                          W_buf[wr], A_buf[wr], M, N, K,
                          block_m, block_n, (k_tile + 1) * TILE_K,
                          mw, kw, ka, na);
        }

        // 计算: 每线程 4×4, 每 lk 一次读 4W+4A 做 16 FMA
        #pragma unroll
        for (int lk = 0; lk < TILE_K; lk++) {
            float w[REG_M], a[REG_N];
            #pragma unroll
            for (int i = 0; i < REG_M; i++)
                w[i] = __half2float(W_buf[rd][m0 + i][lk]);
            #pragma unroll
            for (int j = 0; j < REG_N; j++)
                a[j] = __half2float(A_buf[rd][lk][n0 + j]);
            #pragma unroll
            for (int i = 0; i < REG_M; i++)
                #pragma unroll
                for (int j = 0; j < REG_N; j++)
                    accum[i][j] += w[i] * a[j];
        }

        __syncthreads();
        rd ^= 1;
        wr ^= 1;
    }

    // 写回: 每线程 16 个 FP16
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

__declspec(dllexport) int launch_nvfp4_gemmtc_v3(
    const uint8_t* d_weight_packed,
    const uint8_t* d_weight_scales,
    const __half*  d_activation,
    __half*        d_output,
    int M, int N, int K)
{
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(THREAD_ROWS * THREAD_COLS);  // 256 线程

    nvfp4_gemmtc_v3_kernel<<<grid, block>>>(
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

// v3 shared 占用: W_buf 8KB + A_buf 8KB = 16KB (对比 v2 的 8KB)
__declspec(dllexport) int get_kernel_shared_mem_bytes() {
    return NUM_STAGES * TILE_M * TILE_K * 2   // W_buf
         + NUM_STAGES * TILE_K * TILE_N * 2;  // A_buf
}

}  // extern "C"
