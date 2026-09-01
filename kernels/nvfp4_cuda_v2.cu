/*
 * NVFP4 Fused Dequant-GEMM Kernel — 阶段3 主版本
 * ============================================================
 * 2 stage 双缓冲 + CUDA core FP16 GEMM (无 TensorCore, 降低复杂度)
 * V100 适配: 无 cp.async, 用 __syncthreads() 软件管控多 stage shared memory
 *
 * 设计要点:
 *   1. 双缓冲: __shared__ W_buf[2][TILE_M][TILE_K], A_buf[2][TILE_K][TILE_N]
 *      - rd buffer: 当前 mma 计算源
 *      - wr buffer: 预加载下一块
 *   2. 反量化在 load 阶段完成: global 4bit + scale -> shared FP16
 *      - 每 16 个权重元素 1 个 E8M0 scale, 反量化后存 FP16
 *   3. CUDA core FP16 GEMM: 每个线程负责 output 的一个 (m, n) 元素
 *      - 沿 K 维累加, __half 累加器
 *      - TensorCore mma.sync 留到 2 stage 跑通后再叠加
 *
 * 依赖阶段1/2 成果:
 *   - NVFP4 格式规格 (E2M1 码点, E8M0 scale)
 *   - 反量化逻辑 (与 nvfp4_cuda.cu 一致)
 *   - scale_byte 上限 254 (发现6: 避免 exp2f 溢出)
 *
 * 编译: nvcc -shared -o nvfp4_cuda_v2.dll nvfp4_cuda_v2.cu
 *   (等 MSVC Build Tools 安装后)
 *
 * 验证: nvfp4_cuda_v2_simulate.py (无 GPU 环境用 Python 模拟器验证逻辑)
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdint>

// ============================================================
// 常量定义 (与 nvfp4_cuda.cu, nvfp4_codec.py 一致)
// ============================================================

#define NVFP4_BLOCK_SIZE 16
#define E8M0_BIAS 127

// E2M1 码点绝对值表 (8 个码点, 索引 0-7)
__constant__ float E2M1_CODEPOINTS[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

// Tile 尺寸 (必须 NVFP4_BLOCK_SIZE 的倍数, 阶段3 用 32)
#define TILE_M 32
#define TILE_N 32
#define TILE_K 32

// 双缓冲 stage 数 (D3 对照实验: 默认 2, 编译期 -DNUM_STAGES=3 出 3-stage 对照 dll)
#ifndef NUM_STAGES
#define NUM_STAGES 2
#endif


// ============================================================
// Kernel: NVFP4 Fused Dequant-GEMM (2 stage 双缓冲)
// ============================================================
// 每个 block 计算 output 的一个 [TILE_M × TILE_N] tile
// block 内 (TILE_M × TILE_N) 个线程, 每个线程负责一个 output 元素
//
// 沿 K 维分块迭代:
//   for each K-tile:
//     1. 反量化 weight[tile_m, k_tile] 到 shared W_buf[wr]  (4bit -> FP16)
//     2. 加载 activation[k_tile, tile_n] 到 shared A_buf[wr] (FP16 直传)
//     3. __syncthreads()
//     4. 用 W_buf[rd], A_buf[rd] 做累加 (FP16 GEMM)
//     5. __syncthreads()
//     6. rd ^= 1, wr ^= 1
// D3 通用化: 加载+反量化一个 K-tile 到指定 stage 缓冲 (原 prologue/主循环两份重复代码合并)
__device__ __forceinline__ void load_ktile(
    const uint8_t* __restrict__ weight_packed,
    const uint8_t* __restrict__ weight_scales,
    const __half*  __restrict__ activation,
    int M, int N, int K,
    int k_tile, int buf_idx,
    int block_m, int block_n, int local_m, int local_n,
    __half W_buf[][TILE_M][TILE_K],
    __half A_buf[][TILE_K][TILE_N])
{
    const int k_base = k_tile * TILE_K;
    const int global_n = block_n * TILE_N + local_n;

    // 反量化 weight[global_m_row, k] -> W_buf[buf_idx] (每线程负责 (local_m, lk) 列)
    for (int lk = local_n; lk < TILE_K; lk += TILE_N) {
        int global_k = k_base + lk;
        int global_m_row = block_m * TILE_M + local_m;
        if (global_m_row < M && global_k < K) {
            int pack_idx = global_k / 2;
            uint8_t packed_byte = weight_packed[global_m_row * (K / 2) + pack_idx];
            uint8_t nibble = (global_k % 2 == 0)
                             ? (packed_byte & 0x0F)
                             : ((packed_byte >> 4) & 0x0F);
            uint8_t code = nibble & 0x07;
            float abs_val = E2M1_CODEPOINTS[code];
            uint8_t stored_sign = (nibble >> 3) & 0x01;  // v2: sign 从 nibble bit3 提取 (发现12修复)
            float val = stored_sign ? -abs_val : abs_val;
            int block_idx = global_k / NVFP4_BLOCK_SIZE;
            uint8_t scale_byte = weight_scales[global_m_row * (K / NVFP4_BLOCK_SIZE) + block_idx];
            float scale = exp2f((float)scale_byte - (float)E8M0_BIAS);  // 发现6: sb<=254 防溢出
            W_buf[buf_idx][local_m][lk] = __float2half_rn(val * scale);
        } else {
            W_buf[buf_idx][local_m][lk] = __float2half(0.0f);
        }
    }

    // 加载 activation[k, global_n] -> A_buf[buf_idx] (每线程负责 (lk, local_n))
    for (int lk = local_m; lk < TILE_K; lk += TILE_M) {
        int global_k = k_base + lk;
        if (global_k < K && global_n < N) {
            A_buf[buf_idx][lk][local_n] = activation[global_k * N + global_n];
        } else {
            A_buf[buf_idx][lk][local_n] = __float2half(0.0f);
        }
    }
}

__global__ void nvfp4_fused_dequant_gemm_2stage_kernel(
    const uint8_t* __restrict__ weight_packed,    // [M, K/2]  4bit packed, nibble=[sign|exp|man] (v2: 符号内嵌)
    const uint8_t* __restrict__ weight_scales,    // [M, K/16] E8M0 scale per block
    const __half*  __restrict__ activation,       // [K, N]    FP16 激活
    __half*        __restrict__ output,           // [M, N]    FP16 输出
    int M, int N, int K
) {
    // block 索引
    int block_m = blockIdx.y;  // tile 行索引
    int block_n = blockIdx.x;  // tile 列索引

    // 线程索引 (每个线程负责 output 的一个元素)
    int tid = threadIdx.x;
    int local_m = tid / TILE_N;   // tile 内行索引
    int local_n = tid % TILE_N;   // tile 内列索引

    // 全局 output 坐标
    int global_m = block_m * TILE_M + local_m;
    int global_n = block_n * TILE_N + local_n;

    // 双缓冲 shared memory
    // W_buf[stage][TILE_M][TILE_K]: 反量化后的权重 FP16
    // A_buf[stage][TILE_K][TILE_N]: 激活 FP16 (直传)
    __shared__ __half W_buf[NUM_STAGES][TILE_M][TILE_K];
    __shared__ __half A_buf[NUM_STAGES][TILE_K][TILE_N];

    // 累加器 (float32, 与真实 CUDA kernel 一致; 避免 k_tile 间 FP16 截断误差)
    // 阶段3 不上 TensorCore, CUDA core FP16 GEMM 用 float 累加, 最后转 FP16 输出
    float accum = 0.0f;

    // ceil: 支持任意 K (非 TILE_K 倍数时最后一个 tile 越界部分填 0)
    int num_k_tiles = (K + TILE_K - 1) / TILE_K;

    // ============================================================
    // 预加载前 NUM_STAGES-1 块 (2-stage: tile0; 3-stage: tile0+tile1)
    // ============================================================
    for (int p = 0; p < NUM_STAGES - 1 && p < num_k_tiles; p++) {
        load_ktile(weight_packed, weight_scales, activation, M, N, K,
                   p, p % NUM_STAGES, block_m, block_n, local_m, local_n,
                   W_buf, A_buf);
    }
    if (num_k_tiles > 0) __syncthreads();

    // ============================================================
    // 主循环: S-stage 缓冲 (迭代 k_tile 计算 buf[k_tile%S],
    // 同时预加载 tile k_tile+S-1 到 buf[(k_tile+S-1)%S])
    // 写 buf[(k-1)%S] 与 读 buf[(k-1)%S] (上一迭代) 之间隔本迭代末 sync, 无冲突
    // ============================================================
    for (int k_tile = 0; k_tile < num_k_tiles; k_tile++) {
        int preload = k_tile + NUM_STAGES - 1;
        if (preload < num_k_tiles) {
            load_ktile(weight_packed, weight_scales, activation, M, N, K,
                       preload, preload % NUM_STAGES,
                       block_m, block_n, local_m, local_n, W_buf, A_buf);
        }

        const int rd = k_tile % NUM_STAGES;
        for (int lk = 0; lk < TILE_K; lk++) {
            __half w = W_buf[rd][local_m][lk];
            __half a = A_buf[rd][lk][local_n];
            // FP16 乘 + float32 累加 (CUDA core, 非 TensorCore)
            accum += __half2float(__hmul(w, a));
        }

        __syncthreads();
    }

    // ============================================================
    // 写回 output (float32 累加器 -> FP16 输出)
    // ============================================================
    if (global_m < M && global_n < N) {
        output[global_m * N + global_n] = __float2half_rn(accum);
    }
}


// ============================================================
// Host 函数: 启动 2 stage 双缓冲 kernel (extern "C" 导出, ctypes 可调)
// ============================================================
extern "C" {

__declspec(dllexport) int launch_nvfp4_fused_dequant_gemm_2stage(
    const uint8_t* d_weight_packed,
    const uint8_t* d_weight_scales,
    const __half*  d_activation,
    __half*        d_output,
    int M, int N, int K
) {
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(TILE_M * TILE_N);  // 每线程一个 output 元素 (1024 线程/block)

    nvfp4_fused_dequant_gemm_2stage_kernel<<<grid, block>>>(
        d_weight_packed, d_weight_scales,
        d_activation, d_output,
        M, N, K
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        return (int)err;  // 返回非 0 = kernel 启动失败
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        return (int)err;  // 返回非 0 = 执行失败
    }
    return 0;  // 成功
}

// 工具: 分配 device 内存
__declspec(dllexport) void* gpu_malloc(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);
    if (err != cudaSuccess) return nullptr;
    return ptr;
}

// 工具: 释放 device 内存
__declspec(dllexport) void gpu_free(void* ptr) {
    cudaFree(ptr);
}

// 工具: host -> device 拷贝
__declspec(dllexport) int gpu_memcpy_h2d(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    return (err == cudaSuccess) ? 0 : (int)err;
}

// 工具: device -> host 拷贝
__declspec(dllexport) int gpu_memcpy_d2h(void* dst, const void* src, size_t bytes) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost);
    return (err == cudaSuccess) ? 0 : (int)err;
}

// 工具: 获取 GPU 信息 (与阶段2 nvfp4_cuda.cu 接口一致)
__declspec(dllexport) int get_gpu_info(char* name_buf, int buf_size, int* cc_major, int* cc_minor) {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        return -1;  // 无 GPU
    }

    cudaDeviceProp prop;
    err = cudaGetDeviceProperties(&prop, 0);
    if (err != cudaSuccess) {
        return -2;
    }

    int name_len = 0;
    while (prop.name[name_len] != '\0' && name_len < buf_size - 1) {
        name_buf[name_len] = prop.name[name_len];
        name_len++;
    }
    name_buf[name_len] = '\0';

    *cc_major = prop.major;
    *cc_minor = prop.minor;

    // 顺带打印 shared memory 信息 (阶段3 剖析关注点)
    // 注意: V100 每 SM 96KB shared/L1 可配置, 每 block 上限 48KB (静态)
    return 0;
}

// 工具: 查询 kernel shared memory 占用 (阶段3 剖析关注点)
// 返回: 静态 shared memory 每 block 字节数
__declspec(dllexport) int get_kernel_shared_mem_bytes() {
    // 2 stage 双缓冲:
    //   W_buf[2][32][32] + A_buf[2][32][32] = 4 * 32*32 * 2 bytes(half)
    //   = 4 * 1024 * 2 = 8192 bytes = 8 KB
    int w_buf = NUM_STAGES * TILE_M * TILE_K * 2;   // __half = 2 bytes
    int a_buf = NUM_STAGES * TILE_K * TILE_N * 2;
    return w_buf + a_buf;
}

}  // extern "C"
