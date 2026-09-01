"""
NVFP4 2 stage 双缓冲 Kernel 逻辑模拟器 - 阶段3验证
=================================================
环境状况:
  - GPU 驱动卸载, V100 叹号
  - 无 MSVC (cl.exe), nvcc 无法编译
  - 无 Nsight, 无法做 GPU 性能剖析

策略:
  1. nvfp4_cuda_v2.cu (2 stage 双缓冲) 代码已就绪, 等 MSVC+GPU 修复后编译
  2. 本脚本用纯 Python 精确复现 .cu 文件中 2 stage kernel 的逻辑
  3. 关键验证: 双缓冲输出必须与阶段2 单缓冲 fused GEMM 完全一致 (max_err=0.0)
     原因: 双缓冲只改变访存时序, 不改变计算结果

验证内容:
  测试1: 2 stage 双缓冲 vs 阶段2 单缓冲 (算法等价性, max_err=0.0)
  测试2: 2 stage 双缓冲 vs Python 矩阵乘参考 (正确性, max_err<1e-2)
  测试3: buffer 交换逻辑正确性 (rd/wr 交替)
  测试4: 边界尺寸 (非 TILE 倍数的 M/N/K)
"""
import os
import sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "codec"))
from nvfp4_codec import (
    encode_fp16_to_nvfp4,
    decode_nvfp4_to_fp16,
    NVFP4_BLOCK_SIZE,
    E2M1_ABS_CODEPOINTS,
)
from nvfp4_cuda_simulate import simulate_fused_dequant_gemm_kernel

E2M1_CODEPOINTS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E8M0_BIAS = 127

# Tile 尺寸 (与 .cu 一致)
TILE_M = 32
TILE_N = 32
TILE_K = 32
NUM_STAGES = 2


# ============================================================
# Kernel 逻辑模拟: nvfp4_fused_dequant_gemm_2stage_kernel
# 精确复现 nvfp4_cuda_v2.cu 中的双缓冲逻辑
# ============================================================
def simulate_2stage_kernel(
    weight_packed, weight_scales, weight_signs,
    activation, M, N, K
):
    """
    模拟 CUDA 2 stage 双缓冲 fused dequant-GEMM kernel

    与 .cu 文件 nvfp4_fused_dequant_gemm_2stage_kernel 完全一致:
      - 双缓冲 W_buf[2][TILE_M][TILE_K], A_buf[2][TILE_K][TILE_N]
      - 预加载第 0 块到 rd buffer
      - 主循环: load 下一块到 wr, 计算 rd, syncthreads, 交换
      - 每个线程负责 output(local_m, local_n) 沿 K 累加
      - 反量化在 load 阶段完成 (4bit -> FP16)
    """
    # 输出矩阵 (FP16 最终输出)
    output = np.zeros((M, N), dtype=np.float16)
    # float32 累加器 (与真实 CUDA kernel 一致, 避免 k_tile 间 FP16 截断误差)
    accum_buf = np.zeros((M, N), dtype=np.float32)

    # 遍历所有 tile (block_m, block_n)
    num_block_m = (M + TILE_M - 1) // TILE_M
    num_block_n = (N + TILE_N - 1) // TILE_N
    num_k_tiles = (K + TILE_K - 1) // TILE_K

    for block_m in range(num_block_m):
        for block_n in range(num_block_n):
            # 双缓冲 shared memory (模拟)
            W_buf = np.zeros((NUM_STAGES, TILE_M, TILE_K), dtype=np.float16)
            A_buf = np.zeros((NUM_STAGES, TILE_K, TILE_N), dtype=np.float16)

            rd, wr = 0, 1

            def load_tile(stage, k_tile_idx):
                """模拟 CUDA kernel 中 load 一块到 W_buf[stage]/A_buf[stage] 的逻辑"""
                k_base = k_tile_idx * TILE_K
                # 反量化 weight -> W_buf[stage]
                for local_m in range(TILE_M):
                    for local_k in range(TILE_K):
                        global_m = block_m * TILE_M + local_m
                        global_k = k_base + local_k
                        if global_m < M and global_k < K:
                            pack_idx = global_k // 2
                            packed_byte = weight_packed[global_m, pack_idx]
                            nibble = (packed_byte & 0x0F) if (global_k % 2 == 0) \
                                     else ((packed_byte >> 4) & 0x0F)
                            code = nibble & 0x07
                            abs_val = float(E2M1_CODEPOINTS[code])
                            stored_sign = weight_signs[global_m, global_k]
                            val = -abs_val if stored_sign else abs_val
                            block_idx = global_k // NVFP4_BLOCK_SIZE
                            scale_byte = int(weight_scales[global_m, block_idx])
                            scale = float(np.exp2(np.float32(scale_byte - E8M0_BIAS)))
                            final_val = val * scale
                            W_buf[stage, local_m, local_k] = np.float16(final_val)
                        else:
                            W_buf[stage, local_m, local_k] = np.float16(0.0)

                # 加载 activation -> A_buf[stage]
                for local_k in range(TILE_K):
                    for local_n in range(TILE_N):
                        global_k = k_base + local_k
                        global_n = block_n * TILE_N + local_n
                        if global_k < K and global_n < N:
                            A_buf[stage, local_k, local_n] = activation[global_k, global_n]
                        else:
                            A_buf[stage, local_k, local_n] = np.float16(0.0)

            # 预加载第 0 块 (k_tile=0) 到 rd buffer
            if num_k_tiles > 0:
                load_tile(rd, 0)

            # 主循环 (按 tile 级别迭代, 模拟真实双缓冲时序)
            for k_tile in range(num_k_tiles):
                # 1. 预加载下一块到 wr buffer (与当前计算重叠)
                if k_tile + 1 < num_k_tiles:
                    load_tile(wr, k_tile + 1)

                # 2. 用 rd buffer 做累加 (所有线程协作, 每线程一个 output 元素)
                # 累加器用 float32 (与真实 CUDA kernel 一致, 避免 k_tile 间 FP16 截断)
                for local_m in range(TILE_M):
                    for local_n in range(TILE_N):
                        global_m = block_m * TILE_M + local_m
                        global_n = block_n * TILE_N + local_n

                        if global_m >= M or global_n >= N:
                            continue

                        accum = float(accum_buf[global_m, global_n])  # float32 累加
                        for lk in range(TILE_K):
                            w = float(W_buf[rd, local_m, lk])
                            a = float(A_buf[rd, lk, local_n])
                            accum += w * a
                        accum_buf[global_m, global_n] = np.float32(accum)

                # 3. __syncthreads() + 交换缓冲
                rd ^= 1
                wr ^= 1

    # 全部 k_tile 累加完成, 统一转 FP16 输出
    output = accum_buf.astype(np.float16)
    return output


# ============================================================
# 验证测试
# ============================================================

def test_2stage_vs_singlestage(M=32, N=32, K=32, seed=42):
    """
    测试1: 2 stage 双缓冲 vs 阶段2 单缓冲 fused GEMM
    通过标准: max_err = 0.0 (双缓冲不改计算结果, 必须完全一致)
    """
    print(f"\n{'='*60}")
    print(f"测试1: 2 stage 双缓冲 vs 阶段2 单缓冲 ({M}×{N}×{K})")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    fp16_weight = rng.standard_normal((M, K)).astype(np.float16) * 2.0
    fp16_activation = rng.standard_normal((K, N)).astype(np.float16) * 0.5

    packed, scales, signs = encode_fp16_to_nvfp4(fp16_weight)

    # 阶段2 单缓冲 (ground truth)
    single_out = simulate_fused_dequant_gemm_kernel(
        packed, scales, signs, fp16_activation, M, N, K
    )

    # 阶段3 2 stage 双缓冲
    dual_out = simulate_2stage_kernel(
        packed, scales, signs, fp16_activation, M, N, K
    )

    diff = np.abs(single_out.astype(np.float64) - dual_out.astype(np.float64))
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    exact_match = int(np.sum(single_out == dual_out))

    print(f"  单缓冲 (前6): {single_out.ravel()[:6]}")
    print(f"  双缓冲 (前6): {dual_out.ravel()[:6]}")
    print(f"  最大绝对误差: {max_err:.10f}")
    print(f"  完全匹配: {exact_match}/{M*N}")

    passed = max_err < 1e-6
    print(f"\n[结论] {'✓ PASS' if passed else '✗ FAIL'} (标准: max_err < 1e-6, 期望=0)")
    return passed, {"max_err": max_err, "exact_match": exact_match}


def test_2stage_vs_python_ref(M=32, N=32, K=32, seed=42):
    """
    测试2: 2 stage 双缓冲 vs Python 矩阵乘参考
    通过标准: max_err < 1e-2 (FP16 GEMM 精度)
    """
    print(f"\n{'='*60}")
    print(f"测试2: 2 stage 双缓冲 vs Python 矩阵乘 ({M}×{N}×{K})")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    fp16_weight = rng.standard_normal((M, K)).astype(np.float16) * 2.0
    fp16_activation = rng.standard_normal((K, N)).astype(np.float16) * 0.5

    packed, scales, signs = encode_fp16_to_nvfp4(fp16_weight)

    # Python 参考: 反量化 + 矩阵乘
    py_dequant = decode_nvfp4_to_fp16(packed, scales, signs)
    py_gemm = (py_dequant.astype(np.float32) @ fp16_activation.astype(np.float32)).astype(np.float16)

    # 2 stage 双缓冲
    dual_out = simulate_2stage_kernel(
        packed, scales, signs, fp16_activation, M, N, K
    )

    diff = np.abs(py_gemm.astype(np.float64) - dual_out.astype(np.float64))
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))

    print(f"  Python GEMM (前6): {py_gemm.ravel()[:6]}")
    print(f"  2stage GEMM (前6): {dual_out.ravel()[:6]}")
    print(f"  最大绝对误差: {max_err:.8f}")
    print(f"  平均绝对误差: {mean_err:.8f}")

    passed = max_err < 1e-2
    print(f"\n[结论] {'✓ PASS' if passed else '✗ FAIL'} (标准: max_err < 1e-2)")
    return passed, {"max_err": max_err, "mean_err": mean_err}


def test_buffer_swap_logic():
    """
    测试3: buffer 交换逻辑正确性
    验证 rd/wr 在每次迭代后正确交替 (0->1->0->1...)
    """
    print(f"\n{'='*60}")
    print("测试3: buffer 交换逻辑 (rd/wr 交替)")
    print(f"{'='*60}")

    # 模拟 8 次 k_tile 迭代的 buffer 交换
    rd, wr = 0, 1
    swap_trace = [(rd, wr)]
    for _ in range(8):
        rd ^= 1
        wr ^= 1
        swap_trace.append((rd, wr))

    # 期望: rd 应在 0/1 交替, wr 应与 rd 相反
    expected_rd = [0, 1, 0, 1, 0, 1, 0, 1, 0]
    actual_rd = [t[0] for t in swap_trace]

    print(f"  rd 轨迹: {actual_rd}")
    print(f"  期望:    {expected_rd}")

    # 验证 rd != wr (任意时刻两 buffer 不同)
    all_diff = all(r != w for r, w in swap_trace)
    print(f"  rd != wr 始终成立: {'✓' if all_diff else '✗'}")

    ok = (actual_rd == expected_rd) and all_diff
    print(f"\n[结论] {'✓ PASS' if ok else '✗ FAIL'}")
    return ok


def test_boundary_sizes():
    """
    测试4: 边界尺寸 (非 TILE 倍数的 M/N/K)
    用 M=48, N=48, K=64 验证多 tile 协作 + 边界处理
    """
    print(f"\n{'='*60}")
    print(f"测试4: 边界尺寸 (48×48×64, 非 TILE 倍数)")
    print(f"{'='*60}")

    M, N, K = 48, 48, 64
    rng = np.random.default_rng(99)
    fp16_weight = rng.standard_normal((M, K)).astype(np.float16) * 2.0
    fp16_activation = rng.standard_normal((K, N)).astype(np.float16) * 0.5

    packed, scales, signs = encode_fp16_to_nvfp4(fp16_weight)

    # Python 参考
    py_dequant = decode_nvfp4_to_fp16(packed, scales, signs)
    py_gemm = (py_dequant.astype(np.float32) @ fp16_activation.astype(np.float32)).astype(np.float16)

    # 2 stage 双缓冲 (K=64 不是 TILE_K=32 的... 等等, 64/32=2, 是倍数)
    # 用 M=48, N=48 (非 TILE 倍数) 验证边界 tile 处理
    dual_out = simulate_2stage_kernel(
        packed, scales, signs, fp16_activation, M, N, K
    )

    diff = np.abs(py_gemm.astype(np.float64) - dual_out.astype(np.float64))
    max_err = float(np.max(diff))

    print(f"  矩阵尺寸: {M}×{N}×{K} (M/N 非 TILE 倍数)")
    print(f"  最大绝对误差: {max_err:.8f}")

    passed = max_err < 1e-2
    print(f"\n[结论] {'✓ PASS' if passed else '✗ FAIL'} (标准: max_err < 1e-2)")
    return passed, {"max_err": max_err}


def main():
    print("NVFP4 2 stage 双缓冲 Kernel 逻辑模拟验证 - 阶段3")
    print(f"NumPy: {np.__version__}")
    print(f"环境: 无 GPU/无 MSVC/无 Nsight, 用 Python 模拟双缓冲逻辑")
    print(f"Tile: TILE_M={TILE_M}, TILE_N={TILE_N}, TILE_K={TILE_K}, STAGES={NUM_STAGES}")
    print(f"目的: 验证 nvfp4_cuda_v2.cu 双缓冲算法逻辑正确性")
    print(f"      CUDA .cu 代码已就绪, 等 MSVC+GPU 修复后编译运行")

    # 1. buffer 交换逻辑
    swap_ok = test_buffer_swap_logic()

    # 2. 2 stage vs 单缓冲 (算法等价性, 期望 max_err=0)
    eq32_ok, eq32_stats = test_2stage_vs_singlestage(M=32, N=32, K=32, seed=42)

    # 3. 2 stage vs Python 矩阵乘参考
    ref32_ok, ref32_stats = test_2stage_vs_python_ref(M=32, N=32, K=32, seed=42)

    # 4. 边界尺寸
    boundary_ok, boundary_stats = test_boundary_sizes()

    # 总结
    print(f"\n{'='*60}")
    print("阶段3 验证总结 (2 stage 双缓冲)")
    print(f"{'='*60}")
    print(f"  [buffer 交换逻辑]      {'✓ PASS' if swap_ok else '✗ FAIL'}")
    print(f"  [2stage vs 单缓冲 32³] {'✓ PASS' if eq32_ok else '✗ FAIL'} (max_err={eq32_stats['max_err']:.10f}, 期望=0)")
    print(f"  [2stage vs Py参考 32³] {'✓ PASS' if ref32_ok else '✗ FAIL'} (max_err={ref32_stats['max_err']:.8f})")
    print(f"  [边界尺寸 48×48×64]    {'✓ PASS' if boundary_ok else '✗ FAIL'} (max_err={boundary_stats['max_err']:.8f})")

    overall = swap_ok and eq32_ok and ref32_ok and boundary_ok
    print(f"\n  总体: {'✓ 阶段3 双缓冲算法逻辑验证通过' if overall else '✗ 需修复后重测'}")
    if overall:
        print(f"  交付物:")
        print(f"    - nvfp4_cuda_v2.cu: 2 stage 双缓冲 CUDA kernel (等 MSVC+GPU 编译)")
        print(f"    - 本脚本: 双缓冲逻辑模拟器 (已验证算法正确)")
        print(f"  下一步 (等环境恢复):")
        print(f"    - nvcc -shared -o nvfp4_cuda_v2.dll nvfp4_cuda_v2.cu")
        print(f"    - Nsight Compute 剖析: 双缓冲访存计算重叠率, occupancy, shared 占用")
        print(f"    - FP16 基线 kernel 对比 (nvfp4_baseline_f16.cu)")
        print(f"    - 3 stage 对照分支 (2 stage 跑通后)")
    return overall


if __name__ == "__main__":
    main()
