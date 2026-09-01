"""
阶段3 性能对比: NVFP4 2-stage vs FP16 基线
============================================
同一 tile 结构、同一线程组织的对照实验:
  - nvfp4_cuda_v2.dll: NVFP4 4bit 权重 + 软件反量化 + FP16 GEMM
  - nvfp4_baseline_f16.dll: FP16 权重直传 GEMM (无反量化)

产出指标 (每矩阵尺寸):
  - best/avg 耗时 (ms)
  - 计算吞吐 (TFLOP/s)
  - NVFP4/FP16 性能比 => 反量化净开销
  - 权重显存占用对比 (4bit vs 16bit)
"""
import ctypes
import os
import sys
import subprocess
import time
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "codec"))
from nvfp4_codec import encode_fp16_to_nvfp4

BASE_DIR = os.path.join(_ROOT, "kernels")
V2_DLL = os.path.join(BASE_DIR, "nvfp4_cuda_v2.dll")
V2_CU = os.path.join(BASE_DIR, "nvfp4_cuda_v2.cu")
F16_DLL = os.path.join(BASE_DIR, "nvfp4_baseline_f16.dll")
F16_CU = os.path.join(BASE_DIR, "nvfp4_baseline_f16.cu")


def compile_one(cu_path, dll_path):
    if os.path.exists(dll_path):
        print(f"[编译] 已存在, 跳过: {os.path.basename(dll_path)}")
        return True
    print(f"[编译] {os.path.basename(cu_path)} ...")
    nvcc = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
    ccbin = (r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
             r"\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64")
    r = subprocess.run([nvcc, "-shared", "-o", dll_path, cu_path, "-O3",
                        "-arch=sm_70", "-ccbin", ccbin],
                       capture_output=True, text=True, timeout=180)
    ok = r.returncode == 0 and os.path.exists(dll_path)
    print(f"  {'成功' if ok else '失败: ' + r.stderr[:300]}")
    return ok


def load_and_setup(dll_path):
    lib = ctypes.CDLL(dll_path)
    lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib.gpu_malloc.restype = ctypes.c_void_p
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_h2d.restype = ctypes.c_int
    lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_d2h.restype = ctypes.c_int
    lib.launch_nvfp4_fused_dequant_gemm_2stage.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_nvfp4_fused_dequant_gemm_2stage.restype = ctypes.c_int
    return lib


def load_f16(dll_path):
    lib = ctypes.CDLL(dll_path)
    lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib.gpu_malloc.restype = ctypes.c_void_p
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_h2d.restype = ctypes.c_int
    lib.launch_fp16_gemm_2stage.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_fp16_gemm_2stage.restype = ctypes.c_int
    return lib


def bench_nvfp4(lib, M, N, K, warmup=3, repeat=10):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((M, K)) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)
    packed, scales = encode_fp16_to_nvfp4(w)  # v2: 符号内嵌 nibble, 无独立 signs

    pb, sb, ab = (packed.astype(np.uint8).tobytes(), scales.astype(np.uint8).tobytes(),
                  a.tobytes())
    d_p, d_s, d_a = (lib.gpu_malloc(len(pb)), lib.gpu_malloc(len(sb)),
                     lib.gpu_malloc(len(ab)))
    d_o = lib.gpu_malloc(M * N * 2)
    lib.gpu_memcpy_h2d(d_p, pb, len(pb))
    lib.gpu_memcpy_h2d(d_s, sb, len(sb))
    lib.gpu_memcpy_h2d(d_a, ab, len(ab))

    for _ in range(warmup):
        lib.launch_nvfp4_fused_dequant_gemm_2stage(d_p, d_s, d_a, d_o, M, N, K)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        lib.launch_nvfp4_fused_dequant_gemm_2stage(d_p, d_s, d_a, d_o, M, N, K)
        times.append(time.perf_counter() - t0)

    for p in (d_p, d_s, d_a, d_o):
        lib.gpu_free(p)

    best_ms = min(times) * 1000
    return best_ms, (len(pb) + len(sb))  # 权重字节数 (v2: packed + scales)


def bench_f16(lib, M, N, K, warmup=3, repeat=10):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((M, K)) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N)) * 0.5).astype(np.float16)

    wb, ab = w.tobytes(), a.tobytes()
    d_w, d_a = lib.gpu_malloc(len(wb)), lib.gpu_malloc(len(ab))
    d_o = lib.gpu_malloc(M * N * 2)
    lib.gpu_memcpy_h2d(d_w, wb, len(wb))
    lib.gpu_memcpy_h2d(d_a, ab, len(ab))

    for _ in range(warmup):
        lib.launch_fp16_gemm_2stage(d_w, d_a, d_o, M, N, K)
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        lib.launch_fp16_gemm_2stage(d_w, d_a, d_o, M, N, K)
        times.append(time.perf_counter() - t0)

    for p in (d_w, d_a, d_o):
        lib.gpu_free(p)

    best_ms = min(times) * 1000
    return best_ms, len(wb)  # 权重字节数


def main():
    print("阶段3 性能对比: NVFP4 2-stage 软件反量化 vs FP16 基线")
    print(f"V100 Tesla V100-SXM2-16GB (TCC), 理论 FP16 TensorCore 62.7 TFLOP/s")
    print("(注: 本 kernel 用 CUDA core 非 TensorCore, 实测值远低于理论峰值属预期)")

    if not compile_one(V2_CU, V2_DLL):
        return 1
    if not compile_one(F16_CU, F16_DLL):
        return 1

    lib_nvfp4 = load_and_setup(V2_DLL)
    lib_f16 = load_f16(F16_DLL)

    print(f"\n{'尺寸':<18}{'NVFP4(ms)':<12}{'FP16(ms)':<12}{'比值':<8}"
          f"{'NVFP4(TF)':<11}{'FP16(TF)':<11}{'权重显存':<16}")
    print("-" * 88)

    results = []
    for M, N, K in [(256, 256, 256), (512, 512, 512), (1024, 1024, 1024)]:
        try:
            nv_ms, nv_wbytes = bench_nvfp4(lib_nvfp4, M, N, K)
            f16_ms, f16_wbytes = bench_f16(lib_f16, M, N, K)

            flops = 2.0 * M * N * K
            nv_tf = flops / (nv_ms / 1000) / 1e12
            f16_tf = flops / (f16_ms / 1000) / 1e12
            ratio = nv_ms / f16_ms  # >1 = NVFP4 更慢
            mem_ratio = nv_wbytes / f16_wbytes

            size_str = f"{M}x{N}x{K}"
            print(f"{size_str:<18}{nv_ms:<12.3f}{f16_ms:<12.3f}{ratio:<8.3f}"
                  f"{nv_tf:<11.3f}{f16_tf:<11.3f}{mem_ratio:<16.4f}")
            results.append({"size": size_str, "nv_ms": nv_ms, "f16_ms": f16_ms,
                            "ratio": ratio, "nv_tf": nv_tf, "f16_tf": f16_tf,
                            "mem_ratio": mem_ratio})
        except Exception as e:
            print(f"{M}x{N}x{K}: 失败 - {e}")

    if results:
        print("\n结论:")
        avg_ratio = sum(r["ratio"] for r in results) / len(results)
        print(f"  NVFP4/FP16 平均耗时比: {avg_ratio:.3f} "
              f"(反量化净开销 {(avg_ratio-1)*100:+.1f}%)")
        print(f"  权重显存比: ~{results[0]['mem_ratio']:.4f} (理论 0.3125 = 4bit/16bit + scale)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
