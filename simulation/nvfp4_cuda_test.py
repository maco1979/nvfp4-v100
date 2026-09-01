"""
NVFP4 CUDA Kernel 验证脚本 - 阶段2核心验证
=============================================
验证目标:
  1. CUDA kernel 反量化结果与 Python 参考实现 (nvfp4_codec.py) 逐元素对齐
  2. Fused dequant-GEMM 结果与 Python 等效计算对齐
  3. 通过标准: 最大绝对误差 < 1e-3 (反量化), < 1e-2 (GEMM)

环境处理:
  - 如果 GPU 可用: 编译 DLL -> ctypes 加载 -> 运行 kernel -> 对比
  - 如果 GPU 不可用: 编译验证语法 -> 记录代码 -> 等 GPU 可用时再跑
"""
import ctypes
import os
import sys
import subprocess
import numpy as np

# 导入阶段1 Python 参考实现
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "codec"))
from nvfp4_codec import (
    encode_fp16_to_nvfp4,
    decode_nvfp4_to_fp16,
    scale_byte_to_float,
    NVFP4_BLOCK_SIZE,
)

DLL_PATH = os.path.join(_ROOT, "kernels", "nvfp4_cuda.dll")
CU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvfp4_cuda.cu")


def compile_cuda():
    """用 nvcc 编译 CUDA 代码为 DLL"""
    print("[编译] 使用 nvcc 编译 nvfp4_cuda.cu ...")
    cmd = ["nvcc", "-shared", "-o", DLL_PATH, CU_PATH]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if result.returncode == 0:
            print(f"[编译] ✓ 成功: {DLL_PATH}")
            if os.path.exists(DLL_PATH):
                size_kb = os.path.getsize(DLL_PATH) / 1024
                print(f"       DLL 大小: {size_kb:.1f} KB")
            return True
        else:
            print(f"[编译] ✗ 失败 (返回码 {result.returncode})")
            print(f"       stderr: {result.stderr[:500]}")
            return False
    except FileNotFoundError:
        print("[编译] ✗ nvcc 未找到, 请确认 CUDA toolkit 已安装")
        return False
    except subprocess.TimeoutExpired:
        print("[编译] ✗ 编译超时")
        return False


def load_dll():
    """加载编译好的 DLL"""
    if not os.path.exists(DLL_PATH):
        print(f"[加载] ✗ DLL 不存在: {DLL_PATH}")
        return None
    try:
        lib = ctypes.CDLL(DLL_PATH)
        print(f"[加载] ✓ DLL 加载成功")
        return lib
    except OSError as e:
        print(f"[加载] ✗ DLL 加载失败: {e}")
        return None


def check_gpu(lib):
    """检查 GPU 是否可用"""
    if lib is None:
        return False, "DLL 未加载"

    name_buf = ctypes.create_string_buffer(256)
    cc_major = ctypes.c_int(0)
    cc_minor = ctypes.c_int(0)

    lib.get_gpu_info.argtypes = [
        ctypes.c_char_p, ctypes.c_int,
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)
    ]
    lib.get_gpu_info.restype = ctypes.c_int

    result = lib.get_gpu_info(name_buf, 256, ctypes.byref(cc_major), ctypes.byref(cc_minor))

    if result == 0:
        gpu_name = name_buf.value.decode('utf-8', errors='replace')
        print(f"[GPU] ✓ 检测到: {gpu_name}")
        print(f"      Compute capability: {cc_major.value}.{cc_minor.value}")
        if cc_major.value >= 7:
            print(f"      ✓ 支持 TensorCore (Volta+)")
        else:
            print(f"      ⚠ 不支持 TensorCore (需 Volta+)")
        return True, gpu_name
    elif result == -1:
        print(f"[GPU] ✗ 无 CUDA 设备")
        return False, "无 GPU"
    else:
        print(f"[GPU] ✗ 查询失败 (错误码 {result})")
        return False, f"查询失败 ({result})"


def fp16_to_raw_bytes(fp16_array):
    """FP16 numpy 数组 -> uint16 原始字节 (供 CUDA __half 传输)"""
    return fp16_array.view(np.uint16)


def raw_bytes_to_fp16(uint16_array, shape):
    """uint16 原始字节 -> FP16 numpy 数组"""
    return uint16_array.view(np.float16).reshape(shape)


def test_dequant_alignment(lib, M=16, N=16, seed=42):
    """
    测试1: CUDA 反量化 kernel 与 Python 参考实现对齐
    输入: 随机 FP16 矩阵 -> NVFP4 编码 -> CUDA 反量化 vs Python 反量化
    通过标准: 最大绝对误差 < 1e-3
    """
    print(f"\n{'='*60}")
    print(f"测试1: CUDA 反量化 vs Python 参考实现 ({M}×{N})")
    print(f"{'='*60}")

    # 1. 生成测试数据
    rng = np.random.default_rng(seed)
    fp16_in = rng.standard_normal((M, N)).astype(np.float16) * 2.0
    print(f"[数据] 输入 FP16: shape={fp16_in.shape}, range=[{fp16_in.min():.2f}, {fp16_in.max():.2f}]")

    # 2. Python 参考编码
    packed, scales, signs = encode_fp16_to_nvfp4(fp16_in)
    print(f"[编码] packed: {packed.shape}, scales: {scales.shape}, signs: {signs.shape}")

    # 3. Python 参考反量化 (ground truth)
    py_out = decode_nvfp4_to_fp16(packed, scales, signs)
    print(f"[Python] 反量化输出: shape={py_out.shape}")

    # 4. CUDA 反量化
    # 准备 GPU 内存
    packed_flat = np.ascontiguousarray(packed.ravel().astype(np.uint8))
    scales_flat = np.ascontiguousarray(scales.ravel().astype(np.uint8))
    signs_flat = np.ascontiguousarray(signs.ravel().astype(np.uint8))
    out_raw = np.zeros(M * N, dtype=np.uint16)

    # 分配 GPU 内存
    d_packed = ctypes.c_void_p()
    d_scales = ctypes.c_void_p()
    d_signs = ctypes.c_void_p()
    d_output = ctypes.c_void_p()

    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]
    lib.cudaFree.restype = ctypes.c_int

    cudaMemcpyHostToDevice = 1

    lib.cudaMalloc(ctypes.byref(d_packed), packed_flat.nbytes)
    lib.cudaMalloc(ctypes.byref(d_scales), scales_flat.nbytes)
    lib.cudaMalloc(ctypes.byref(d_signs), signs_flat.nbytes)
    lib.cudaMalloc(ctypes.byref(d_output), out_raw.nbytes)

    lib.cudaMemcpy(d_packed, packed_flat.ctypes.data, packed_flat.nbytes, cudaMemcpyHostToDevice)
    lib.cudaMemcpy(d_scales, scales_flat.ctypes.data, scales_flat.nbytes, cudaMemcpyHostToDevice)
    lib.cudaMemcpy(d_signs, signs_flat.ctypes.data, signs_flat.nbytes, cudaMemcpyHostToDevice)

    # 启动 kernel
    lib.launch_nvfp4_dequant.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int
    ]
    lib.launch_nvfp4_dequant.restype = None
    lib.launch_nvfp4_dequant(d_packed, d_scales, d_signs, d_output, M, N)

    # 拷回结果
    lib.cudaMemcpy(out_raw.ctypes.data, d_output, out_raw.nbytes, 2)  # DeviceToHost=2

    # 释放 GPU 内存
    lib.cudaFree(d_packed)
    lib.cudaFree(d_scales)
    lib.cudaFree(d_signs)
    lib.cudaFree(d_output)

    # 转换为 FP16
    cuda_out = raw_bytes_to_fp16(out_raw, (M, N))
    print(f"[CUDA] 反量化输出: shape={cuda_out.shape}")

    # 5. 对比
    py_arr = py_out.astype(np.float64)
    cuda_arr = cuda_out.astype(np.float64)
    abs_diff = np.abs(py_arr - cuda_arr)
    max_err = float(np.max(abs_diff))
    mean_err = float(np.mean(abs_diff))

    print(f"\n[对齐分析]")
    print(f"  Python 输出 (前4个): {py_out.ravel()[:4]}")
    print(f"  CUDA   输出 (前4个): {cuda_out.ravel()[:4]}")
    print(f"  最大绝对误差: {max_err:.8f}")
    print(f"  平均绝对误差: {mean_err:.8f}")
    print(f"  完全匹配元素数: {np.sum(py_arr == cuda_arr)}/{M*N}")

    passed = max_err < 1e-3
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n[结论] {status} (通过标准: max_err < 1e-3)")
    return passed, {"max_err": max_err, "mean_err": mean_err}


def test_fused_gemm_alignment(lib, M=16, N=16, K=16, seed=42):
    """
    测试2: CUDA fused dequant-GEMM 与 Python 等效计算对齐
    计算: C(M,K) = dequant_NVFP4(A(M,N)) × B(N,K)
    通过标准: 最大绝对误差 < 1e-2 (GEMM 累积误差稍大)
    """
    print(f"\n{'='*60}")
    print(f"测试2: CUDA fused dequant-GEMM vs Python ({M}×{N} × {N}×{K})")
    print(f"{'='*60}")

    # 1. 生成测试数据
    rng = np.random.default_rng(seed)
    fp16_weight = rng.standard_normal((M, N)).astype(np.float16) * 2.0  # 权重矩阵
    fp16_activation = rng.standard_normal((N, K)).astype(np.float16) * 0.5  # 激活矩阵
    print(f"[数据] 权重: {fp16_weight.shape}, 激活: {fp16_activation.shape}")

    # 2. Python 参考: 先反量化再矩阵乘
    packed, scales, signs = encode_fp16_to_nvfp4(fp16_weight)
    py_dequant = decode_nvfp4_to_fp16(packed, scales, signs)
    py_gemm = (py_dequant.astype(np.float32) @ fp16_activation.astype(np.float32)).astype(np.float16)
    print(f"[Python] GEMM 输出: shape={py_gemm.shape}")

    # 3. CUDA fused dequant-GEMM
    packed_flat = np.ascontiguousarray(packed.ravel().astype(np.uint8))
    scales_flat = np.ascontiguousarray(scales.ravel().astype(np.uint8))
    signs_flat = np.ascontiguousarray(signs.ravel().astype(np.uint8))
    act_raw = fp16_to_raw_bytes(np.ascontiguousarray(fp16_activation))
    out_raw = np.zeros(M * K, dtype=np.uint16)

    lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    lib.cudaMalloc.restype = ctypes.c_int
    lib.cudaMemcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    lib.cudaMemcpy.restype = ctypes.c_int
    lib.cudaFree.argtypes = [ctypes.c_void_p]

    d_wp = ctypes.c_void_p(); d_ws = ctypes.c_void_p(); d_wsg = ctypes.c_void_p()
    d_act = ctypes.c_void_p(); d_out = ctypes.c_void_p()

    lib.cudaMalloc(ctypes.byref(d_wp), packed_flat.nbytes)
    lib.cudaMalloc(ctypes.byref(d_ws), scales_flat.nbytes)
    lib.cudaMalloc(ctypes.byref(d_wsg), signs_flat.nbytes)
    lib.cudaMalloc(ctypes.byref(d_act), act_raw.nbytes)
    lib.cudaMalloc(ctypes.byref(d_out), out_raw.nbytes)

    lib.cudaMemcpy(d_wp, packed_flat.ctypes.data, packed_flat.nbytes, 1)
    lib.cudaMemcpy(d_ws, scales_flat.ctypes.data, scales_flat.nbytes, 1)
    lib.cudaMemcpy(d_wsg, signs_flat.ctypes.data, signs_flat.nbytes, 1)
    lib.cudaMemcpy(d_act, act_raw.ctypes.data, act_raw.nbytes, 1)

    lib.launch_nvfp4_fused_dequant_gemm.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.launch_nvfp4_fused_dequant_gemm.restype = None
    lib.launch_nvfp4_fused_dequant_gemm(d_wp, d_ws, d_wsg, d_act, d_out, M, N, K)

    lib.cudaMemcpy(out_raw.ctypes.data, d_out, out_raw.nbytes, 2)

    lib.cudaFree(d_wp); lib.cudaFree(d_ws); lib.cudaFree(d_wsg)
    lib.cudaFree(d_act); lib.cudaFree(d_out)

    cuda_gemm = raw_bytes_to_fp16(out_raw, (M, K))
    print(f"[CUDA] GEMM 输出: shape={cuda_gemm.shape}")

    # 4. 对比
    py_arr = py_gemm.astype(np.float64)
    cuda_arr = cuda_gemm.astype(np.float64)
    abs_diff = np.abs(py_arr - cuda_arr)
    max_err = float(np.max(abs_diff))
    mean_err = float(np.mean(abs_diff))
    rel_err = abs_diff / (np.abs(py_arr) + 1e-6)

    print(f"\n[对齐分析]")
    print(f"  Python GEMM (前4个): {py_gemm.ravel()[:4]}")
    print(f"  CUDA   GEMM (前4个): {cuda_gemm.ravel()[:4]}")
    print(f"  最大绝对误差: {max_err:.6f}")
    print(f"  平均绝对误差: {mean_err:.6f}")
    print(f"  平均相对误差: {float(np.mean(rel_err))*100:.4f}%")

    passed = max_err < 1e-2
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"\n[结论] {status} (通过标准: max_err < 1e-2)")
    return passed, {"max_err": max_err, "mean_err": mean_err}


def main():
    print("NVFP4 CUDA Kernel 验证 - 阶段2")
    print(f"NumPy: {np.__version__}")
    print(f"工作目录: {os.path.dirname(os.path.abspath(__file__))}")

    # 1. 编译 CUDA
    if not compile_cuda():
        print("\n[跳过] CUDA 编译失败, 阶段2 无法运行 GPU 验证")
        print("[记录] CUDA kernel 代码已就绪 (nvfp4_cuda.cu), 等 GPU 环境可用时编译运行")
        return False

    # 2. 加载 DLL
    lib = load_dll()
    if lib is None:
        return False

    # 3. 检查 GPU
    gpu_ok, gpu_info = check_gpu(lib)
    if not gpu_ok:
        print(f"\n[跳过] GPU 不可用 ({gpu_info}), 无法运行 kernel 验证")
        print("[记录] CUDA kernel 代码 + DLL 编译成功, 等 GPU 可用时运行验证")
        return False

    # 4. 测试1: 反量化对齐
    deq16_ok, deq16_stats = test_dequant_alignment(lib, M=16, N=16, seed=42)
    deq32_ok, deq32_stats = test_dequant_alignment(lib, M=32, N=32, seed=7)

    # 5. 测试2: fused dequant-GEMM 对齐
    gemm16_ok, gemm16_stats = test_fused_gemm_alignment(lib, M=16, N=16, K=16, seed=42)
    gemm32_ok, gemm32_stats = test_fused_gemm_alignment(lib, M=32, N=32, K=32, seed=7)

    # 总结
    print(f"\n{'='*60}")
    print("阶段2 验证总结")
    print(f"{'='*60}")
    print(f"  [反量化 16×16]  {'✓ PASS' if deq16_ok else '✗ FAIL'} (max_err={deq16_stats['max_err']:.8f})")
    print(f"  [反量化 32×32]  {'✓ PASS' if deq32_ok else '✗ FAIL'} (max_err={deq32_stats['max_err']:.8f})")
    print(f"  [GEMM 16×16×16] {'✓ PASS' if gemm16_ok else '✗ FAIL'} (max_err={gemm16_stats['max_err']:.6f})")
    print(f"  [GEMM 32×32×32] {'✓ PASS' if gemm32_ok else '✗ FAIL'} (max_err={gemm32_stats['max_err']:.6f})")

    overall = deq16_ok and deq32_ok and gemm16_ok and gemm32_ok
    print(f"\n  总体: {'✓ 阶段2 通过, 可进入阶段3 (TensorCore + 双缓冲)' if overall else '✗ 需修复后重测'}")
    return overall


if __name__ == "__main__":
    main()
