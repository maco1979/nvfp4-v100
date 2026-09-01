"""
NVFP4 Kernel v5 验证脚本 — TensorCore wmma (阶段4 优化第3档)
====================================================================
1. GPU 信息 + v5 shared 占用 (预期 36KB, v4 为 18KB)
2. 真值比对: v5 GPU vs Python 参考
   - TC 内建 FP32 累加, 精度预期回到 v3 水平 (优于 v4 两级累加)
   - 验收 (发现24 标准): mean_rel < 0.01 且 max_abs < max(0.1, 1%×输出幅值)
3. v4 vs v5 同输入一致性
4. 性能三方对比: v4 / v5 / FP16基线, LTX 主力尺寸 (目标 v5/v4 >= 2x)
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
V5_DLL = os.path.join(BASE, "nvfp4_cuda_v5.dll")
V5_CU = os.path.join(BASE, "nvfp4_cuda_v5.cu")
V4_DLL = os.path.join(BASE, "nvfp4_cuda_v4.dll")
F16_DLL = os.path.join(BASE, "nvfp4_baseline_f16.dll")

NVCC = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin\nvcc.exe"
CCBIN = (r"C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
         r"\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64")


def compile_v5():
    if os.path.exists(V5_DLL):
        print(f"[编译] 已存在: {os.path.basename(V5_DLL)}")
        return True
    print("[编译] nvfp4_cuda_v5.cu ...")
    r = subprocess.run([NVCC, "-shared", "-o", V5_DLL, V5_CU, "-O3",
                        "-arch=sm_70", "-ccbin", CCBIN],
                       capture_output=True, timeout=300)
    err = r.stderr.decode("gbk", errors="replace") if r.stderr else ""
    ok = r.returncode == 0 and os.path.exists(V5_DLL)
    print(f"  {'成功' if ok else '失败: ' + err[-800:]}")
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
    """真值比对: GPU v5 vs Python 参考 (发现24 相对标准)"""
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


def v4_v5_consistency(launch_v4, launch_v5, M, N, K, seed):
    """v4 vs v5 同输入逐元素比对"""
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)
    packed, scales = encode_fp16_to_nvfp4(w)
    o4 = run_gemm(LIB, launch_v4, w, packed, scales, a, M, N, K)
    o5 = run_gemm(LIB, launch_v5, w, packed, scales, a, M, N, K)
    d = np.abs(o4.astype(np.float32) - o5.astype(np.float32))
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
    if not compile_v5():
        return 1
    LIB = ctypes.CDLL(V5_DLL)
    launch_v5 = setup_nvfp4(LIB, "launch_nvfp4_gemmtc_v5")

    name_buf = ctypes.create_string_buffer(256)
    cc_m, cc_n = ctypes.c_int(0), ctypes.c_int(0)
    LIB.get_gpu_info(name_buf, 256, ctypes.byref(cc_m), ctypes.byref(cc_n))
    print(f"[GPU] {name_buf.value.decode()} (CC {cc_m.value}.{cc_n.value})")
    print(f"[v5 shared] {LIB.get_kernel_shared_mem_bytes()/1024:.1f} KB/block "
          f"(v4: 18.0 KB, wmma 16x16x16, TILE 64x64x64, 8 warps 2x4)")

    # ---- 真值比对 ----
    print("\n=== 真值比对: v5 vs Python 参考 ===")
    cases = [
        (64, 64, 64, "tile 整倍数"),
        (128, 64, 32, "tile 整倍数 2"),
        (100, 100, 112, "边界: M/N/K 均非 tile 倍数 (K 为 16 倍数)"),
        (512, 256, 512, "中等"),
        (1024, 1024, 1024, "大 (与阶段3对齐)"),
        (512, 256, 16384, "长 K: TC FP32 累加精度 (256 个 k-tile)"),
    ]
    all_pass = True
    for M, N, K, note in cases:
        mx, mr, thr = truth_test(launch_v5, M, N, K, seed=42)
        ok = mx < thr and mr < 0.01
        all_pass &= ok
        print(f"  {M}x{N}x{K} [{note}]: max_abs={mx:.4f} (thr={thr:.3f}) "
              f"mean_rel={mr:.4f} {'PASS' if ok else 'FAIL'}")
    if not all_pass:
        print("真值比对 FAIL, 终止性能测试")
        return 1

    # ---- v4 vs v5 一致性 ----
    print("\n=== v4 vs v5 同输入一致性 (v5 应更接近 v3 水平) ===")
    launch_v4 = setup_nvfp4(ctypes.CDLL(V4_DLL), "launch_nvfp4_gemmtc_v4")
    for M, N, K in [(100, 100, 112), (512, 256, 512), (512, 256, 16384)]:
        mx, mn = v4_v5_consistency(launch_v4, launch_v5, M, N, K, seed=7)
        print(f"  {M}x{N}x{K}: max_diff={mx:.4f} mean_diff={mn:.5f}")

    # ---- 性能三方对比 (LTX 主力尺寸) ----
    lib_f16 = ctypes.CDLL(F16_DLL)
    lib_f16.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib_f16.gpu_malloc.restype = ctypes.c_void_p
    lib_f16.gpu_free.argtypes = [ctypes.c_void_p]
    lib_f16.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib_f16.launch_fp16_gemm_2stage.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib_f16.launch_fp16_gemm_2stage.restype = ctypes.c_int

    print("\n=== 性能对比: v4 / v5 / FP16基线 (LTX 主力尺寸) ===")
    header = (f"{'M':>6} {'K':>6} {'T':>6} | {'v4(ms)':>9} {'v5(ms)':>9} "
              f"{'FP16(ms)':>9} | {'v5/v4':>6} {'v5/FP16':>7} | {'v5(TF)':>7}")
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
        t4 = bench(launch_v4, M, T, K, warm, rep)
        t5 = bench(launch_v5, M, T, K, warm, rep)
        tf = bench_f16(lib_f16.launch_fp16_gemm_2stage, M, T, K, warm, rep)
        v5_tf = flops / (t5 / 1000) / 1e12
        print(f"{M:>6} {K:>6} {T:>6} | {t4:>9.3f} {t5:>9.3f} {tf:>9.3f} | "
              f"{t5/t4:>6.3f} {t5/tf:>7.3f} | {v5_tf:>7.3f}")
        rows.append({"M": M, "K": K, "T": T, "cnt": cnt, "v4_ms": t4, "v5_ms": t5,
                     "f16_ms": tf})

    print("\n结论:")
    sp = [r["v4_ms"] / r["v5_ms"] for r in rows]
    print(f"  v5/v4 加速比: min={min(sp):.2f}x max={max(sp):.2f}x "
          f"mean={sum(sp)/len(sp):.2f}x (目标 >=2x)")
    best_tf = max(2.0 * r['M'] * r['T'] * r['K'] / (r['v5_ms'] / 1000) / 1e12
                  for r in rows)
    print(f"  v5 峰值吞吐: {best_tf:.2f} TF (V100 TensorCore FP16 峰值 125 TF)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
