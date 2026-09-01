/*
 * FP16 原生 GEMM 基线 Kernel - 阶段3 性能对比基准
 * ============================================================
 * 与 nvfp4_cuda_v2.cu 的 2 stage 双缓冲 NVFP4 kernel 做对照:
 *   - 相同 tile 尺寸 (32x32x32), 相同双缓冲结构, 相同线程组织
 *   - 唯一区别: 权重直接以 FP16 存储 (无 4bit 反量化)
 *
 * 目的: 隔离"NVFP4 软件反量化"的净开销
 *   性能(NVFP4) vs 性能(FP16基线) => 反量化开销占比
 *
 * 编译: nvcc -shared -o nvfp4_baseline_f16.dll nvfp4_baseline_f16.cu
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

#define TILE_M 32
#define TILE_N 32
#define TILE_K 32
#define NUM_STAGES 2


// ============================================================
// Kernel: FP16 GEMM (2 stage 双缓冲, 与 v2 结构完全一致)
// ============================================================
__global__ void fp16_gemm_2stage_kernel(
    const __half* __restrict__ weight,    // [M, K] FP16 (未量化)
    const __half* __restrict__ activation, // [K, N] FP16
    __half*       __restrict__ output,     // [M, N] FP16
    int M, int N, int K
) {
    int block_m = blockIdx.y;
    int block_n = blockIdx.x;

    int tid = threadIdx.x;
    int local_m = tid / TILE_N;
    int local_n = tid % TILE_N;

    int global_m = block_m * TILE_M + local_m;
    int global_n = block_n * TILE_N + local_n;

    __shared__ __half W_buf[NUM_STAGES][TILE_M][TILE_K];
    __shared__ __half A_buf[NUM_STAGES][TILE_K][TILE_N];

    float accum = 0.0f;  // float32 累加器 (与 v2 一致)

    int num_k_tiles = (K + TILE_K - 1) / TILE_K;
    int rd = 0, wr = 1;

    // 预加载第 0 块
    if (num_k_tiles > 0) {
        // FP16 直传 (无反量化 - 这是与 v2 的唯一区别)
        for (int lk = local_n; lk < TILE_K; lk += TILE_N) {
            int gk = lk;
            int gm = block_m * TILE_M + local_m;
            W_buf[rd][local_m][lk] = (gm < M && gk < K)
                ? weight[gm * K + gk]
                : __float2half(0.0f);
        }
        for (int lk = local_m; lk < TILE_K; lk += TILE_M) {
            int gk = lk;
            A_buf[rd][lk][local_n] = (gk < K && global_n < N)
                ? activation[gk * N + global_n]
                : __float2half(0.0f);
        }
        __syncthreads();
    }

    // 主循环: 双缓冲
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        if (k_tile + 1 < num_k_tiles) {
            int next_k_base = (k_tile + 1) * TILE_K;
            for (int lk = local_n; lk < TILE_K; lk += TILE_N) {
                int gk = next_k_base + lk;
                int gm = block_m * TILE_M + local_m;
                W_buf[wr][local_m][lk] = (gm < M && gk < K)
                    ? weight[gm * K + gk]
                    : __float2half(0.0f);
            }
            for (int lk = local_m; lk < TILE_K; lk += TILE_M) {
                int gk = next_k_base + lk;
                A_buf[wr][lk][local_n] = (gk < K && global_n < N)
                    ? activation[gk * N + global_n]
                    : __float2half(0.0f);
            }
        }

        for (int lk = 0; lk < TILE_K; lk++) {
            __half w = W_buf[rd][local_m][lk];
            __half a = A_buf[rd][lk][local_n];
            accum += __half2float(__hmul(w, a));
        }

        __syncthreads();
        rd ^= 1;
        wr ^= 1;
    }

    if (global_m < M && global_n < N) {
        output[global_m * N + global_n] = __float2half_rn(accum);
    }
}


// ============================================================
// extern "C" 导出 (与 nvfp4_cuda_v2.dll 接口风格一致)
// ============================================================
extern "C" {

__declspec(dllexport) int launch_fp16_gemm_2stage(
    const __half* d_weight,
    const __half* d_activation,
    __half*       d_output,
    int M, int N, int K
) {
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(TILE_M * TILE_N);

    fp16_gemm_2stage_kernel<<<grid, block>>>(
        d_weight, d_activation, d_output, M, N, K
    );

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

__declspec(dllexport) void gpu_free(void* ptr) {
    cudaFree(ptr);
}

__declspec(dllexport) int gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    return (cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess) ? 0 : -1;
}

__declspec(dllexport) int gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    return (cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost) == cudaSuccess) ? 0 : -1;
}

__declspec(dllexport) int get_gpu_info(char* name_buf, int buf_size, int* cc_major, int* cc_minor) {
    int device_count;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) return -1;
    cudaDeviceProp prop;
    if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) return -2;
    int name_len = 0;
    while (prop.name[name_len] != '\0' && name_len < buf_size - 1) {
        name_buf[name_len] = prop.name[name_len];
        name_len++;
    }
    name_buf[name_len] = '\0';
    *cc_major = prop.major;
    *cc_minor = prop.minor;
    return 0;
}

}  // extern "C"
