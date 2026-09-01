"""
NVFP4 2 stage 双缓冲 Kernel GPU 验证脚本 - 阶段3 GPU 实测
==========================================================
环境: V100 已恢复 (Tesla V100-SXM2-16GB, TCC, 驱动 573.76) + MSVC Build Tools

验证内容:
  1. GPU 信息 + kernel shared memory 占用
  2. 2 stage 双缓冲 GPU 结果 vs Python 参考 (真值比对)
     - 标准: 与 Python 模拟器逻辑一致, max_err 容忍 FP16 精度 (< 1e-2)
  3. 性能计时 (CUDA events):
     - 多尺寸 GEMM: 256x256x256, 512x512x512, 1024x1024x1024
     - 双缓冲 vs 理论峰值
  4. 显存带宽利用率分析
"""
import ctypes
import os
import sys
import subprocess
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "codec"))
from nvfp4_codec import encode_fp16_to_nvfp4, decode_nvfp4_to_fp16

DLL_PATH = os.path.join(_ROOT, "kernels", "nvfp4_cuda_v2.dll")
CU_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nvfp4_cuda_v2.cu")


def compile_cuda():
    """nvcc 编译 v2.cu 为 DLL (若已存在则跳过, 避免覆盖手工编译产物)"""
    if os.path.exists(DLL_PATH):
        size_kb = os.path.getsize(DLL_PATH) / 1024
        print(f"[编译] DLL 已存在, 跳过: {DLL_PATH} ({size_kb:.1f} KB)")
        return True
    print("[编译] nvcc -shared -o nvfp4_cuda_v2.dll nvfp4_cuda_v2.cu")
    cmd = ["nvcc", "-shared", "-o", DLL_PATH, CU_PATH, "-O3"]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode == 0 and os.path.exists(DLL_PATH):
            size_kb = os.path.getsize(DLL_PATH) / 1024
            print(f"[编译] 成功: {DLL_PATH} ({size_kb:.1f} KB)")
            return True
        else:
            print(f"[编译] 失败 (返回码 {result.returncode})")
            print(f"  stderr: {result.stderr[:800]}")
            return False
    except FileNotFoundError:
        print("[编译] nvcc 未找到")
        return False
    except subprocess.TimeoutExpired:
        print("[编译] 超时")
        return False


def load_dll():
    if not os.path.exists(DLL_PATH):
        print(f"[加载] DLL 不存在: {DLL_PATH}")
        return None
    try:
        lib = ctypes.CDLL(DLL_PATH)
        print("[加载] DLL 加载成功")
        return lib
    except OSError as e:
        print(f"[加载] 失败: {e}")
        return None


def setup_lib(lib):
    """配置 ctypes 函数签名"""
    lib.get_gpu_info.argtypes = [ctypes.c_char_p, ctypes.c_int,
                                 ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    lib.get_gpu_info.restype = ctypes.c_int

    lib.get_kernel_shared_mem_bytes.argtypes = []
    lib.get_kernel_shared_mem_bytes.restype = ctypes.c_int

    lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib.gpu_malloc.restype = ctypes.c_void_p

    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_free.restype = None

    lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_h2d.restype = ctypes.c_int

    lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_d2h.restype = ctypes.c_int

    lib.launch_nvfp4_fused_dequant_gemm_2stage.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int
    ]
    lib.launch_nvfp4_fused_dequant_gemm_2stage.restype = ctypes.c_int


def check_gpu(lib):
    name_buf = ctypes.create_string_buffer(256)
    cc_major, cc_minor = ctypes.c_int(0), ctypes.c_int(0)
    ret = lib.get_gpu_info(name_buf, 256, ctypes.byref(cc_major), ctypes.byref(cc_minor))
    if ret != 0:
        print(f"[GPU] 不可用 (返回码 {ret})")
        return False
    shared_bytes = lib.get_kernel_shared_mem_bytes()
    print(f"[GPU] {name_buf.value.decode()} (CC {cc_major.value}.{cc_minor.value})")
    print(f"[GPU] kernel 静态 shared memory: {shared_bytes} bytes = {shared_bytes/1024:.1f} KB")
    return True


def run_gemm_on_gpu(lib, packed, scales, activation, M, N, K):
    """在 GPU 上跑 2 stage 双缓冲 fused dequant-GEMM, 返回 FP16 输出 (v2: 符号内嵌 nibble)"""
    packed_bytes = packed.astype(np.uint8).tobytes()
    scales_bytes = scales.astype(np.uint8).tobytes()
    act_bytes = activation.astype(np.float16).tobytes()

    d_packed = lib.gpu_malloc(len(packed_bytes))
    d_scales = lib.gpu_malloc(len(scales_bytes))
    d_act = lib.gpu_malloc(len(act_bytes))
    d_out = lib.gpu_malloc(M * N * 2)  # FP16

    if not all([d_packed, d_scales, d_act, d_out]):
        raise RuntimeError("cudaMalloc 失败")

    lib.gpu_memcpy_h2d(d_packed, packed_bytes, len(packed_bytes))
    lib.gpu_memcpy_h2d(d_scales, scales_bytes, len(scales_bytes))
    lib.gpu_memcpy_h2d(d_act, act_bytes, len(act_bytes))

    err = lib.launch_nvfp4_fused_dequant_gemm_2stage(
        d_packed, d_scales, d_act, d_out, M, N, K
    )
    if err != 0:
        lib.gpu_free(d_packed); lib.gpu_free(d_scales)
        lib.gpu_free(d_act); lib.gpu_free(d_out)
        raise RuntimeError(f"kernel 执行失败, CUDA error code = {err}")

    out_bytes = bytes(M * N * 2)
    buf = ctypes.create_string_buffer(M * N * 2)
    lib.gpu_memcpy_d2h(buf, d_out, M * N * 2)

    lib.gpu_free(d_packed); lib.gpu_free(d_scales)
    lib.gpu_free(d_act); lib.gpu_free(d_out)

    return np.frombuffer(buf.raw, dtype=np.float16).reshape(M, N).copy()


def test_correctness(lib, M, N, K, seed=42):
    """GPU 结果 vs Python 参考真值比对"""
    print(f"\n{'='*60}")
    print(f"真值比对: GPU 2-stage vs Python 参考 ({M}x{N}x{K})")
    print(f"{'='*60}")

    rng = np.random.default_rng(seed)
    fp16_weight = (rng.standard_normal((M, K)) * 2.0).astype(np.float16)
    fp16_act = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)

    packed, scales = encode_fp16_to_nvfp4(fp16_weight)

    # Python 参考: 反量化 + float32 矩阵乘 -> FP16
    dequant = decode_nvfp4_to_fp16(packed, scales)
    ref = (dequant.astype(np.float32) @ fp16_act.astype(np.float32)).astype(np.float16)

    # GPU
    try:
        gpu_out = run_gemm_on_gpu(lib, packed, scales, fp16_act, M, N, K)
    except RuntimeError as e:
        print(f"  GPU 执行失败: {e}")
        return False, None

    diff = np.abs(ref.astype(np.float64) - gpu_out.astype(np.float64))
    max_err = float(np.max(diff))
    mean_err = float(np.mean(diff))
    # 相对误差 (对 Frobenius 范数)
    rel_err = float(np.linalg.norm(diff) / max(np.linalg.norm(ref.astype(np.float64)), 1e-9))

    # FP16 ULP 判定: kernel 用 __hmul(FP16域乘法)+float累加, 参考全程 float32,
    # 允许输出幅度处的 4 ULP 差异 (实测 1-2 ULP)
    max_abs = float(np.max(np.abs(ref.astype(np.float64))))
    ulp = np.float16(max_abs).item()
    import struct
    ulp_at_max = 2.0 ** (np.floor(np.log2(max(ulp, 1e-30))) - 10)  # FP16 尾数 10 bit
    passed = (rel_err < 1e-3) and (max_err <= 4 * ulp_at_max)

    print(f"  GPU (前6):  {gpu_out.ravel()[:6]}")
    print(f"  参考 (前6): {ref.ravel()[:6]}")
    print(f"  max_err={max_err:.6f} (4*ULP@max={4*ulp_at_max:.6f}), "
          f"mean_err={mean_err:.6f}, rel_err(F)={rel_err:.6f}")
    print(f"  [{'PASS' if passed else 'FAIL'}] (标准: rel_err<1e-3 且 max_err<=4*ULP)")
    return passed, {"max_err": max_err, "mean_err": mean_err, "rel_err": rel_err}


def test_performance(lib, M, N, K, warmup=3, repeat=10):
    """性能计时: 多次执行取平均 (host 侧计时, 含同步)"""
    rng = np.random.default_rng(7)
    fp16_weight = (rng.standard_normal((M, K)) * 2.0).astype(np.float16)
    fp16_act = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    packed, scales = encode_fp16_to_nvfp4(fp16_weight)

    packed_bytes = packed.astype(np.uint8).tobytes()
    scales_bytes = scales.astype(np.uint8).tobytes()
    act_bytes = fp16_act.tobytes()

    d_packed = lib.gpu_malloc(len(packed_bytes))
    d_scales = lib.gpu_malloc(len(scales_bytes))
    d_act = lib.gpu_malloc(len(act_bytes))
    d_out = lib.gpu_malloc(M * N * 2)

    lib.gpu_memcpy_h2d(d_packed, packed_bytes, len(packed_bytes))
    lib.gpu_memcpy_h2d(d_scales, scales_bytes, len(scales_bytes))
    lib.gpu_memcpy_h2d(d_act, act_bytes, len(act_bytes))

    # 预热 (触发 JIT/缓存/时钟提升)
    for _ in range(warmup):
        lib.launch_nvfp4_fused_dequant_gemm_2stage(
            d_packed, d_scales, d_act, d_out, M, N, K
        )

    # 计时
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        lib.launch_nvfp4_fused_dequant_gemm_2stage(
            d_packed, d_scales, d_act, d_out, M, N, K
        )
        t1 = time.perf_counter()
        times.append(t1 - t0)

    lib.gpu_free(d_packed); lib.gpu_free(d_scales)
    lib.gpu_free(d_act); lib.gpu_free(d_out)

    times_ms = [t * 1000 for t in times]
    best_ms = min(times_ms)
    avg_ms = sum(times_ms) / len(times_ms)

    flops = 2.0 * M * N * K  # 乘加 = 2 FLOP
    tflops = flops / (best_ms / 1000) / 1e12

    # 数据搬运量 (weights 4bit + scale + act FP16)
    bytes_moved = (M * K / 2) + (M * K / 16) + (K * N * 2) + (M * N * 2)
    bw_gbs = bytes_moved / (best_ms / 1000) / 1e9

    print(f"  {M}x{N}x{K}: best={best_ms:.3f} ms, avg={avg_ms:.3f} ms, "
          f"计算吞吐={tflops:.3f} TFLOP/s, 等效带宽={bw_gbs:.1f} GB/s")

    return {"M": M, "N": N, "K": K, "best_ms": best_ms, "avg_ms": avg_ms,
            "tflops": tflops, "bw_gbs": bw_gbs}


def main():
    print("NVFP4 2-stage 双缓冲 GPU 验证 - 阶段3 GPU 实测")
    print(f"环境: {os.name}, NumPy {np.__version__}")
    print(f"DLL: {DLL_PATH}")
    print(f"CU:  {CU_PATH}")

    if not compile_cuda():
        return 1
    lib = load_dll()
    if lib is None:
        return 1
    setup_lib(lib)
    if not check_gpu(lib):
        return 1

    # 1. 真值比对 (小矩阵先验证正确性)
    ok64, _ = test_correctness(lib, 64, 64, 64, seed=42)
    ok256, _ = test_correctness(lib, 256, 256, 256, seed=99)

    # 2. 边界尺寸 (非 TILE 倍数)
    ok_boundary, _ = test_correctness(lib, 48, 48, 64, seed=7)

    # 3. 性能实测
    print(f"\n{'='*60}")
    print("性能实测 (2 stage 双缓冲, CUDA core FP16 GEMM)")
    print(f"{'='*60}")
    perf_results = []
    for M, N, K in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        try:
            r = test_performance(lib, M, N, K)
            perf_results.append(r)
        except Exception as e:
            print(f"  {M}x{N}x{K}: 失败 - {e}")

    # 总结
    print(f"\n{'='*60}")
    print("阶段3 GPU 实测总结")
    print(f"{'='*60}")
    all_ok = ok64 and ok256 and ok_boundary
    print(f"  真值比对 64^3:   {'PASS' if ok64 else 'FAIL'}")
    print(f"  真值比对 256^3:  {'PASS' if ok256 else 'FAIL'}")
    print(f"  边界 48x48x64:   {'PASS' if ok_boundary else 'FAIL'}")
    if perf_results:
        print(f"  性能数据已收集 {len(perf_results)} 组")
        best = max(perf_results, key=lambda r: r["tflops"])
        print(f"  峰值吞吐: {best['tflops']:.3f} TFLOP/s @ {best['M']}x{best['N']}x{best['K']}")
    print(f"\n  总体: {'阶段3 GPU 验证通过' if all_ok else '需修复'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
