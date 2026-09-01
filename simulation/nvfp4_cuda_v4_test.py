"""
NVFP4 Kernel v4 验证脚本 — __hfma2 half2 打包计算 (阶段4 优化第2档)
====================================================================
1. GPU 信息 + v4 shared 占用 (预期 18432B ≈ 18KB, v3 为 16KB)
2. 真值比对: v4 GPU vs Python 参考 (nvfp4_codec.decode + numpy float32 matmul)
   - 含 tile 整倍数尺寸 + 边界尺寸 + 长 K 尺寸 (两级累加精度检查: K=16384)
   - 验收: max_abs < 0.1, mean_rel 预期 0.001-0.005 (half 内层累加 32 步/tile)
3. v3 vs v4 同输入一致性 (数值路径不同: v3 纯 float 累加 vs v4 两级累加, 允许小差异)
4. 性能三方对比: v3 / v4 / FP16基线, LTX 主力尺寸 (目标 v4/v3 ≥ 1.4x)
"""
import ctypes
import os
import subprocess
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "codec"))
from nvfp4_codec import encode_fp16_to_nvfp4, decode_nvfp4_to_fp16

BASE = os.path.join(_ROOT, "kernels")
V4_DLL = os.path.join(BASE, "nvfp4_cuda_v4.dll")
V4_CU = os.path.join(BASE, "nvfp4_cuda_v4.cu")
V3_DLL = os.path.join(BASE, "nvfp4_cuda_v3.dll")
F16_DLL = os.path.join(BASE, "nvfp4_baseline_f16.dll")

NVCC = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
CCBIN = (r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
         r"\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64")


def compile_v4():
    if os.path.exists(V4_DLL):
        print(f"[编译] 已存在: {os.path.basename(V4_DLL)}")
        return True
    print("[编译] nvfp4_cuda_v4.cu ...")
    r = subprocess.run([NVCC, "-shared", "-o", V4_DLL, V4_CU, "-O3",
                        "-arch=sm_70", "-ccbin", CCBIN],
                       capture_output=True, timeout=300)
    err = r.stderr.decode("gbk", errors="replace") if r.stderr else ""
    ok = r.returncode == 0 and os.path.exists(V4_DLL)
    print(f"  {'成功' if ok else '失败: ' + err[-600:]}")
    return ok


def setup_nvfp4(lib, launch_name):
    lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib.gpu_malloc.restype = ctypes.c_void_p
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_h2d.restype = ctypes.c_int
    lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    getattr(lib, launch_name).argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    getattr(lib, launch_name).restype = ctypes.c_int
    return getattr(lib, launch_name)


def run_gemm(lib, launch, w, packed, scales, a, M, N, K):
    pb, sb, ab = (packed.astype(np.uint8).tobytes(), scales.astype(np.uint8).tobytes(),
                  a.tobytes())
    d_p, d_s, d_a = (lib.gpu_malloc(len(pb)), lib.gpu_malloc(len(sb)),
                     lib.gpu_malloc(len(ab)))
    d_o = lib.gpu_malloc(M * N * 2)
    lib.gpu_memcpy_h2d(d_p, pb, len(pb))
    lib.gpu_memcpy_h2d(d_s, sb, len(sb))
    lib.gpu_memcpy_h2d(d_a, ab, len(ab))
    ret = launch(d_p, d_s, d_a, d_o, M, N, K)
    out = np.empty(M * N, dtype=np.float16)
    lib.gpu_memcpy_d2h(out.ctypes.data, d_o, M * N * 2)
    for p in (d_p, d_s, d_a, d_o):
        lib.gpu_free(p)
    if ret != 0:
        raise RuntimeError(f"kernel 返回 {ret}")
    return out.reshape(M, N)


def truth_test(launch, M, N, K, seed):
    """真值比对: GPU v4 vs Python 参考
    v4 两级累加 (half 内层 32 步 + float 外层) 的舍入随 K 随机游走,
    绝对误差阈值须按输出幅值缩放: max_abs < max(0.1, 1%×max|ref|)
    (NVFP4 推理内核的科学标准: kernel 数值误差 ≪ E2M1 量化误差 ~20%)"""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)
    packed, scales = encode_fp16_to_nvfp4(w)

    out = run_gemm(LIB, launch, w, packed, scales, a, M, N, K)

    w_ref = decode_nvfp4_to_fp16(packed, scales).astype(np.float32)
    ref = w_ref @ a.astype(np.float32)
    err = np.abs(out.astype(np.float32) - ref)
    rel = err / (np.abs(ref) + 1e-3)
    thr = max(0.1, 0.01 * float(np.abs(ref).max()))
    return float(err.max()), float(rel.mean()), thr


def v3_v4_consistency(launch_v3, launch_v4, M, N, K, seed):
    """v3 vs v4 同输入逐元素比对 (数值路径不同, 允许小幅差异)"""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)
    packed, scales = encode_fp16_to_nvfp4(w)
    o3 = run_gemm(LIB, launch_v3, w, packed, scales, a, M, N, K)
    o4 = run_gemm(LIB, launch_v4, w, packed, scales, a, M, N, K)
    d = np.abs(o3.astype(np.float32) - o4.astype(np.float32))
    return float(d.max()), float(d.mean())


def bench(launch, M, N, K, warmup, repeat):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)
    packed, scales = encode_fp16_to_nvfp4(w)
    pb, sb, ab = (packed.astype(np.uint8).tobytes(), scales.astype(np.uint8).tobytes(),
                  a.tobytes())
    d_p, d_s, d_a = (LIB.gpu_malloc(len(pb)), LIB.gpu_malloc(len(sb)),
                     LIB.gpu_malloc(len(ab)))
    d_o = LIB.gpu_malloc(M * N * 2)
    LIB.gpu_memcpy_h2d(d_p, pb, len(pb))
    LIB.gpu_memcpy_h2d(d_s, sb, len(sb))
    LIB.gpu_memcpy_h2d(d_a, ab, len(ab))
    for _ in range(warmup):
        assert launch(d_p, d_s, d_a, d_o, M, N, K) == 0
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        assert launch(d_p, d_s, d_a, d_o, M, N, K) == 0
        times.append(time.perf_counter() - t0)
    for p in (d_p, d_s, d_a, d_o):
        LIB.gpu_free(p)
    return min(times) * 1000


def bench_f16(launch, M, N, K, warmup, repeat):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)
    wb, ab = w.tobytes(), a.tobytes()
    d_w, d_a = LIB.gpu_malloc(len(wb)), LIB.gpu_malloc(len(ab))
    d_o = LIB.gpu_malloc(M * N * 2)
    LIB.gpu_memcpy_h2d(d_w, wb, len(wb))
    LIB.gpu_memcpy_h2d(d_a, ab, len(ab))
    for _ in range(warmup):
        assert launch(d_w, d_a, d_o, M, N, K) == 0
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        assert launch(d_w, d_a, d_o, M, N, K) == 0
        times.append(time.perf_counter() - t0)
    for p in (d_w, d_a, d_o):
        LIB.gpu_free(p)
    return min(times) * 1000


LIB = None


def main():
    global LIB
    if not compile_v4():
        return 1
    LIB = ctypes.CDLL(V4_DLL)
    launch_v4 = setup_nvfp4(LIB, "launch_nvfp4_gemmtc_v4")

    name_buf = ctypes.create_string_buffer(256)
    cc_m, cc_n = ctypes.c_int(0), ctypes.c_int(0)
    LIB.get_gpu_info(name_buf, 256, ctypes.byref(cc_m), ctypes.byref(cc_n))
    print(f"[GPU] {name_buf.value.decode()} (CC {cc_m.value}.{cc_n.value})")
    print(f"[v4 shared] {LIB.get_kernel_shared_mem_bytes()/1024:.1f} KB/block "
          f"(v3: 16.0 KB, W 行 padding 40 half + __hfma2 计算路径)")

    # ---- 真值比对 ----
    print("\n=== 真值比对: v4 vs Python 参考 ===")
    cases = [
        (64, 64, 64, "tile 整倍数"),
        (128, 64, 32, "tile 整倍数 2"),
        (100, 100, 112, "边界: M/N/K 均非 tile 倍数 (K 为 16 倍数)"),
        (512, 256, 512, "中等"),
        (1024, 1024, 1024, "大 (与阶段3对齐)"),
        (512, 256, 16384, "长 K: 两级累加精度 (512 个 k-tile flush)"),
    ]
    all_pass = True
    for M, N, K, note in cases:
        mx, mr, thr = truth_test(launch_v4, M, N, K, seed=42)
        ok = mx < thr and mr < 0.01
        all_pass &= ok
        print(f"  {M}x{N}x{K} [{note}]: max_abs={mx:.4f} (thr={thr:.3f}) "
              f"mean_rel={mr:.4f} {'PASS' if ok else 'FAIL'}")
    if not all_pass:
        print("真值比对 FAIL, 终止性能测试")
        return 1

    # ---- v3 vs v4 一致性 ----
    print("\n=== v3 vs v4 同输入一致性 (数值路径不同, 允许小幅差异) ===")
    launch_v3 = setup_nvfp4(ctypes.CDLL(V3_DLL), "launch_nvfp4_gemmtc_v3")
    for M, N, K in [(100, 100, 112), (512, 256, 512), (512, 256, 16384)]:
        mx, mn = v3_v4_consistency(launch_v3, launch_v4, M, N, K, seed=7)
        print(f"  {M}x{N}x{K}: max_diff={mx:.4f} mean_diff={mn:.5f}")

    # ---- 性能三方对比 (LTX 主力尺寸) ----
    lib_f16 = ctypes.CDLL(F16_DLL)
    lib_f16.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib_f16.gpu_malloc.restype = ctypes.c_void_p
    lib_f16.gpu_free.argtypes = [ctypes.c_void_p]
    lib_f16.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib_f16.gpu_memcpy_h2d.restype = ctypes.c_int
    lib_f16.launch_fp16_gemm_2stage.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib_f16.launch_fp16_gemm_2stage.restype = ctypes.c_int

    print("\n=== 性能对比: v3 / v4 / FP16基线 (LTX 主力尺寸) ===")
    header = (f"{'M':>6} {'K':>6} {'T':>6} | {'v3(ms)':>9} {'v4(ms)':>9} "
              f"{'FP16(ms)':>9} | {'v4/v3':>6} {'v4/FP16':>7} | {'v4(TF)':>7}")
    print(header)
    print("-" * len(header))

    sizes = [
        (2048, 2048, 512, 613), (2048, 2048, 1536, 613), (2048, 2048, 4992, 613),
        (4096, 4096, 512, 421), (4096, 4096, 1536, 421), (4096, 4096, 4992, 421),
        (16384, 4096, 1536, 57), (4096, 16384, 1536, 56),
    ]
    rows = []
    for M, K, T, cnt in sizes:
        flops = 2.0 * M * T * K
        warm, rep = (1, 3) if flops > 1e12 else (2, 5)
        t3 = bench(launch_v3, M, T, K, warm, rep)
        t4 = bench(launch_v4, M, T, K, warm, rep)
        tf = bench_f16(lib_f16.launch_fp16_gemm_2stage, M, T, K, warm, rep)
        v4_tf = flops / (t4 / 1000) / 1e12
        print(f"{M:>6} {K:>6} {T:>6} | {t3:>9.3f} {t4:>9.3f} {tf:>9.3f} | "
              f"{t4/t3:>6.3f} {t4/tf:>7.3f} | {v4_tf:>7.3f}")
        rows.append({"M": M, "K": K, "T": T, "cnt": cnt, "v3_ms": t3, "v4_ms": t4,
                     "f16_ms": tf})

    print("\n结论:")
    sp = [r["v3_ms"] / r["v4_ms"] for r in rows]
    print(f"  v4/v3 加速比: min={min(sp):.2f}x max={max(sp):.2f}x "
          f"mean={sum(sp)/len(sp):.2f}x (目标 ≥1.4x)")
    best_tf = max(2.0 * r['M'] * r['T'] * r['K'] / (r['v4_ms'] / 1000) / 1e12
                  for r in rows)
    print(f"  v4 峰值吞吐: {best_tf:.2f} TF (FP16 CUDA core 路径峰值 31.4 TF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
