/*
 * NVFP4 CUDA Kernel - 阶段2核心交付物
 * ============================================
 * 研究目标: 在 V100 上软件模拟 Blackwell NVFP4 4-bit 浮点格式
 * 本文件: CUDA kernel 实现 NVFP4 反量化 + fused dequant-GEMM
 *
 * 知识库依据:
 *   - cezanne/operating_systems: CUDA and GPU Computing (Volta/Ampere/Hopper 架构)
 *   - cezanne/programming_languages: PyTorch Deep Learning Framework (GPU 加速栈)
 *   - galileo/algebra: 矩阵基础 (m×n 数表, 乘法列=行)
 *   - NVFP4 格式: 知识库无条目, 模型原生知识 + CPHYSJEPA 物理一致性
 *
 * V100 硬件约束 (豆包方案修正):
 *   1. V100 无 cp.async 指令 (Ampere 才引入)
 *   2. V100 TensorCore 只认 FP16, 4bit 无法直接进乘法阵列
 *   3. 本阶段先用 CUDA core 做 FP16 GEMM, TensorCore 优化留到阶段3
 *
 * 编译: nvcc -shared -o nvfp4_cuda.dll nvfp4_cuda.cu
 *        nvcc -shared -o nvfp4_cuda.so nvfp4_cuda.cu (Linux)
 */

#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>

// ============================================================
// E2M1 码点表 (绝对值, constant memory 加速访问)
// 索引 = 3-bit 编码 [exp(2)][m(1)]
// 0: +0 (subnormal, exp=0,m=0)
// 1: 0.5 (subnormal, exp=0,m=1)
// 2: 1.0 (1.0 × 2^0)
// 3: 1.5 (1.1 × 2^0)
// 4: 2.0 (1.0 × 2^1)
// 5: 3.0 (1.1 × 2^1)
// 6: 4.0 (1.0 × 2^2)
// 7: 6.0 (1.1 × 2^2)
// ============================================================
__constant__ float E2M1_CODEPOINTS[8] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f
};

#define NVFP4_BLOCK_SIZE 16
#define E8M0_BIAS 127

// ============================================================
// Kernel 1: NVFP4 反量化 (4bit -> FP16)
// ============================================================
// 输入:
//   packed_4bit: (M, N/2) uint8, 每字节存2个FP4元素
//   block_scales: (M, N/16) uint8, 每16元素1个E8M0 scale
//   signs: (M, N) uint8, 符号位 (0=正, 1=负)
// 输出:
//   fp16_out: (M, N) __half, 反量化后的FP16矩阵
//
// 每个线程处理一个元素
// 验证标准: 与 nvfp4_codec.py 的 decode_nvfp4_to_fp16 逐元素对齐 (容差 1e-3)
__global__ void nvfp4_dequant_kernel(
    const uint8_t* __restrict__ packed_4bit,
    const uint8_t* __restrict__ block_scales,
    const uint8_t* __restrict__ signs,
    __half* __restrict__ fp16_out,
    int M, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx >= total) return;

    int row = idx / N;
    int col = idx % N;

    // 1. 从 packed_4bit 解包 4-bit nibble
    //    打包格式: 偶数索引在低4bit, 奇数索引在高4bit
    int pack_idx = col / 2;
    uint8_t packed = packed_4bit[row * (N / 2) + pack_idx];
    uint8_t nibble;
    if (col % 2 == 0) {
        nibble = packed & 0x0F;        // 低 4 bit
    } else {
        nibble = (packed >> 4) & 0x0F; // 高 4 bit
    }

    // 2. 提取 3-bit code (低 3 bit)
    //    注: 符号位在 Python 参考实现中单独存储, 这里从 nibble 提取
    //    4-bit 布局: [sign(1)][exp(2)][m(1)] = sign<<3 | code_3bit
    uint8_t sign_bit = (nibble >> 3) & 0x01;
    uint8_t code = nibble & 0x07;

    // 3. 查表得到 E2M1 绝对值
    float abs_val = E2M1_CODEPOINTS[code];

    // 4. 应用符号 (signs 数组优先, 与 Python 实现一致)
    uint8_t stored_sign = signs[idx];
    float val = stored_sign ? -abs_val : abs_val;

    // 5. 读取 block scale 并反归一化
    //    E8M0: scale = 2^(byte - 127)
    int block_idx = col / NVFP4_BLOCK_SIZE;
    uint8_t scale_byte = block_scales[row * (N / NVFP4_BLOCK_SIZE) + block_idx];

    // 用 exp2f 计算 2^(scale_byte - 127), float 精度足够 (scale 是 2 的幂次)
    float scale = exp2f((float)scale_byte - (float)E8M0_BIAS);
    float final_val = val * scale;

    // 6. 转 FP16 输出
    fp16_out[idx] = __float2half_rn(final_val);
}

// ============================================================
// Kernel 2: Fused Dequant-GEMM (NVFP4 权重 × FP16 激活)
// ============================================================
// 计算: C(M,K) = dequant_NVFP4(A(M,N)) × B(N,K)
// A 是 NVFP4 编码的权重矩阵, B 是 FP16 激活矩阵
//
// 本阶段用 CUDA core 实现 (非 TensorCore), 验证算法正确性
// 阶段3 再用 mma.sync TensorCore 优化
//
// 每个线程计算 C 的一个元素
__global__ void nvfp4_fused_dequant_gemm_kernel(
    const uint8_t* __restrict__ weight_packed,    // (M, N/2) NVFP4 权重
    const uint8_t* __restrict__ weight_scales,     // (M, N/16) E8M0 scale
    const uint8_t* __restrict__ weight_signs,      // (M, N) 符号位
    const __half* __restrict__ activation,         // (N, K) FP16 激活
    __half* __restrict__ output,                   // (M, K) FP16 输出
    int M, int N, int K
) {
    // 每个 thread 计算 output(row, col)
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= K) return;

    // 累加器用 float 避免精度损失
    float accum = 0.0f;

    // 遍历 N 维度, 每个 block(16元素) 批量处理
    for (int block_idx = 0; block_idx < N / NVFP4_BLOCK_SIZE; block_idx++) {
        // 读取这个 block 的 scale
        uint8_t scale_byte = weight_scales[row * (N / NVFP4_BLOCK_SIZE) + block_idx];
        float scale = exp2f((float)scale_byte - (float)E8M0_BIAS);

        // 处理 block 内 16 个元素
        int base_col = block_idx * NVFP4_BLOCK_SIZE;
        for (int i = 0; i < NVFP4_BLOCK_SIZE; i++) {
            int n = base_col + i;

            // 解包 NVFP4 权重元素
            int pack_idx = n / 2;
            uint8_t packed = weight_packed[row * (N / 2) + pack_idx];
            uint8_t nibble;
            if (n % 2 == 0) {
                nibble = packed & 0x0F;
            } else {
                nibble = (packed >> 4) & 0x0F;
            }

            uint8_t code = nibble & 0x07;
            uint8_t stored_sign = weight_signs[row * N + n];
            float abs_val = E2M1_CODEPOINTS[code];
            float w_val = stored_sign ? -abs_val : abs_val;
            w_val = w_val * scale;  // 反归一化

            // 读取激活值
            float a_val = __half2float(activation[n * K + col]);

            // 累加
            accum += w_val * a_val;
        }
    }

    // 写入输出 (转 FP16)
    output[row * K + col] = __float2half_rn(accum);
}

// ============================================================
// Host 函数: 启动反量化 kernel
// ============================================================
extern "C" {

void launch_nvfp4_dequant(
    const uint8_t* d_packed,
    const uint8_t* d_scales,
    const uint8_t* d_signs,
    __half* d_output,
    int M, int N
) {
    int total = M * N;
    int block_size = 256;
    int grid_size = (total + block_size - 1) / block_size;

    nvfp4_dequant_kernel<<<grid_size, block_size>>>(
        d_packed, d_scales, d_signs, d_output, M, N
    );
    cudaDeviceSynchronize();
}

void launch_nvfp4_fused_dequant_gemm(
    const uint8_t* d_weight_packed,
    const uint8_t* d_weight_scales,
    const uint8_t* d_weight_signs,
    const __half* d_activation,
    __half* d_output,
    int M, int N, int K
) {
    dim3 block(16, 16);
    dim3 grid((K + block.x - 1) / block.x, (M + block.y - 1) / block.y);

    nvfp4_fused_dequant_gemm_kernel<<<grid, block>>>(
        d_weight_packed, d_weight_scales, d_weight_signs,
        d_activation, d_output, M, N, K
    );
    cudaDeviceSynchronize();
}

// ============================================================
// 工具函数: 获取 GPU 信息
// ============================================================
int get_gpu_info(char* name_buf, int buf_size, int* cc_major, int* cc_minor) {
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

    // 安全拷贝设备名
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
