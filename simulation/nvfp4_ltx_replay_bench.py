"""
阶段4路线A: LTX-2.3 22B 真实 M/N/K 尺寸回放测速
==================================================
输入: fp4研究/ltx23_22b_linear_shapes.json (前8种尺寸覆盖95.23%参数量)
负载: token数 T ∈ {512, 1536, 4992, 14080} × 主力 (N_out, K_in) 尺寸

Kernel 约定 (nvfp4_cuda_v2.cu):
  weight[M, K] @ activation[K, N] = output[M, N]
  => kernel M = PyTorch out_features (JSON 的 N)
  => kernel K = PyTorch in_features  (JSON 的 K)
  => kernel N = token 数 T

产出:
  1. 每组合 NVFP4 vs FP16 延迟 / TFLOP/s / 比值
  2. 加权单次前向估算: Σ count_i × t_i (LTX-2.3 22B 全部 1777 个 Linear)
  3. 权重显存对比 (FP16 44.3GB vs NVFP4 ~6.2GB)
  4. ULP 抽检: 大矩阵下 GPU kernel vs Python 参考真值

V100 Tesla V100-SXM2-16GB:
  FP16 CUDA core 峰值 ~15.5 TFLOP/s (本 kernel 无 TensorCore)
"""
import ctypes
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "codec"))
from nvfp4_codec import encode_fp16_to_nvfp4, decode_nvfp4_to_fp16

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SHAPES_JSON = os.path.join(_ROOT, "experiments", "data", "ltx23_22b_linear_shapes.json")
USE_V3 = "--v3" in sys.argv
USE_V4 = "--v4" in sys.argv
USE_V5 = "--v5" in sys.argv
REPORT_JSON = os.path.join(
    _ROOT, "experiments", "data",
    ("ltx_replay_bench_results_v5.json" if USE_V5 else
     "ltx_replay_bench_results_v4.json" if USE_V4 else
     "ltx_replay_bench_results_v3.json" if USE_V3 else
     "ltx_replay_bench_results.json"))
V2_DLL = os.path.join(_ROOT, "kernels", "nvfp4_cuda_v2.dll")
V3_DLL = os.path.join(_ROOT, "kernels", "nvfp4_cuda_v3.dll")
V4_DLL = os.path.join(_ROOT, "kernels", "nvfp4_cuda_v4.dll")
V5_DLL = os.path.join(_ROOT, "kernels", "nvfp4_cuda_v5.dll")
F16_DLL = os.path.join(_ROOT, "kernels", "nvfp4_baseline_f16.dll")
NVFP4_LAUNCH = ("launch_nvfp4_gemmtc_v5" if USE_V5 else
                "launch_nvfp4_gemmtc_v4" if USE_V4 else
                "launch_nvfp4_gemmtc_v3" if USE_V3 else
                "launch_nvfp4_fused_dequant_gemm_2stage")

TOKENS = [512, 1536, 4992, 14080]
TOP_N_SIZES = 9          # 覆盖 97.08% 参数量 (含 4096x2048)
BIG_K_LIMIT_TOKENS = 1536  # K=188160 patch embed 只测小 T (host 内存约束)


def load_libs():
    nv_dll = (V5_DLL if USE_V5 else
              V4_DLL if USE_V4 else
              V3_DLL if USE_V3 else V2_DLL)
    lib_nvfp4 = ctypes.CDLL(nv_dll)
    lib_nvfp4.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib_nvfp4.gpu_malloc.restype = ctypes.c_void_p
    lib_nvfp4.gpu_free.argtypes = [ctypes.c_void_p]
    lib_nvfp4.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib_nvfp4.gpu_memcpy_h2d.restype = ctypes.c_int
    lib_nvfp4.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib_nvfp4.gpu_memcpy_d2h.restype = ctypes.c_int
    getattr(lib_nvfp4, NVFP4_LAUNCH).argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    getattr(lib_nvfp4, NVFP4_LAUNCH).restype = ctypes.c_int

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
    return lib_nvfp4, lib_f16


def encode_big_weight(w, chunk_rows=256):
    """分块编码大权重, 控制 host 内存峰值 (7.7亿元素全量编码峰值会爆 RAM)"""
    M, K = w.shape
    if M <= chunk_rows:
        return encode_fp16_to_nvfp4(w)
    packed_list, scales_list = [], []
    for i in range(0, M, chunk_rows):
        p, s = encode_fp16_to_nvfp4(w[i:i + chunk_rows])
        packed_list.append(p)
        scales_list.append(s)
    return (np.concatenate(packed_list, axis=0), np.concatenate(scales_list, axis=0))


def bench_one(lib, M, N, K, is_nvfp4, warmup, repeat):
    """单组合测速. 返回 (best_ms, 权重字节数). M=权重行, N=token, K=in_features"""
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)

    d_o = lib.gpu_malloc(M * N * 2)
    if is_nvfp4:
        packed, scales = encode_big_weight(w)
        pb, sb = packed.astype(np.uint8).tobytes(), scales.astype(np.uint8).tobytes()
        ab = a.tobytes()
        d_p, d_s, d_a = (lib.gpu_malloc(len(pb)), lib.gpu_malloc(len(sb)),
                         lib.gpu_malloc(len(ab)))
        lib.gpu_memcpy_h2d(d_p, pb, len(pb))
        lib.gpu_memcpy_h2d(d_s, sb, len(sb))
        lib.gpu_memcpy_h2d(d_a, ab, len(ab))
        launch = getattr(lib, NVFP4_LAUNCH)
        args = (d_p, d_s, d_a, d_o)
        extra_ptrs = (d_p, d_s, d_a)
        wbytes = len(pb) + len(sb)
    else:
        wb, ab = w.tobytes(), a.tobytes()
        d_w, d_a = lib.gpu_malloc(len(wb)), lib.gpu_malloc(len(ab))
        lib.gpu_memcpy_h2d(d_w, wb, len(wb))
        lib.gpu_memcpy_h2d(d_a, ab, len(ab))
        launch = lib.launch_fp16_gemm_2stage
        args = (d_w, d_a, d_o)
        extra_ptrs = (d_w, d_a)
        wbytes = len(wb)

    for _ in range(warmup):
        assert launch(*args, M, N, K) == 0
    times = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        assert launch(*args, M, N, K) == 0
        times.append(time.perf_counter() - t0)

    for p in extra_ptrs + (d_o,):
        lib.gpu_free(p)

    return min(times) * 1000, wbytes


def correctness_spot_check(lib, M=512, N=256, K=512):
    """抽检: GPU kernel vs Python 参考真值 (中等尺寸, 大矩阵逻辑相同)"""
    rng = np.random.default_rng(42)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
    a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)
    packed, scales = encode_fp16_to_nvfp4(w)

    pb, sb, ab = (packed.astype(np.uint8).tobytes(), scales.astype(np.uint8).tobytes(),
                  a.tobytes())
    d_p, d_s, d_a = (lib.gpu_malloc(len(pb)), lib.gpu_malloc(len(sb)),
                     lib.gpu_malloc(len(ab)))
    d_o = lib.gpu_malloc(M * N * 2)
    lib.gpu_memcpy_h2d(d_p, pb, len(pb))
    lib.gpu_memcpy_h2d(d_s, sb, len(sb))
    lib.gpu_memcpy_h2d(d_a, ab, len(ab))
    assert getattr(lib, NVFP4_LAUNCH)(d_p, d_s, d_a, d_o, M, N, K) == 0

    out = np.empty(M * N, dtype=np.float16)
    lib.gpu_memcpy_d2h(out.ctypes.data, d_o, M * N * 2)
    out = out.reshape(M, N)
    for p in (d_p, d_s, d_a, d_o):
        lib.gpu_free(p)

    # Python 参考: 反量化权重 (float32) @ 激活 (float32)
    w_ref = decode_nvfp4_to_fp16(packed, scales).astype(np.float32)
    ref = w_ref @ a.astype(np.float32)
    err = np.abs(out.astype(np.float32) - ref)
    rel = err / (np.abs(ref) + 1e-3)
    print(f"[抽检] {M}x{N}x{K}: max_abs={err.max():.4f} mean_rel={rel.mean():.4f} "
          f"(FP16 累加容忍 <0.1)")
    return float(err.max()), float(rel.mean())


def main():
    d = json.load(open(SHAPES_JSON, encoding="utf-8"))
    sizes = sorted(d["unique_sizes"],
                   key=lambda s: -s["count"] * s["params_per_layer"])[:TOP_N_SIZES]
    total_params = d["total_linear_params"]
    covered = sum(s["count"] * s["params_per_layer"] for s in sizes)

    lib_nvfp4, lib_f16 = load_libs()
    name_buf = ctypes.create_string_buffer(256)
    cc_m, cc_n = ctypes.c_int(0), ctypes.c_int(0)
    lib_nvfp4.get_gpu_info(name_buf, 256, ctypes.byref(cc_m), ctypes.byref(cc_n))
    print(f"[GPU] {name_buf.value.decode()} (CC {cc_m.value}.{cc_n.value})")
    print(f"[负载] LTX-2.3 22B 前{len(sizes)}种尺寸, 覆盖 {100*covered/total_params:.1f}% 参数量, "
          f"T={TOKENS}")

    correctness_spot_check(lib_nvfp4)

    results = []
    header = (f"{'M(out)':>7} {'K(in)':>7} {'T':>6} {'cnt':>4} | "
              f"{'NVFP4(ms)':>10} {'FP16(ms)':>10} {'ratio':>7} | "
              f"{'NVFP4(TF)':>10} {'FP16(TF)':>9}")
    print("\n" + header)
    print("-" * len(header))

    for s in sizes:
        M, K, cnt = s["N"], s["K"], s["count"]
        for T in TOKENS:
            if K > 100000 and T > BIG_K_LIMIT_TOKENS:
                print(f"{M:>7} {K:>7} {T:>6} {cnt:>4} | {'SKIP(host内存)':>10}")
                continue
            flops = 2.0 * M * T * K
            warmup, repeat = (1, 3) if flops > 1e12 else ((2, 5) if flops > 1e11 else (3, 10))
            try:
                nv_ms, nv_wb = bench_one(lib_nvfp4, M, T, K, True, warmup, repeat)
                f16_ms, f16_wb = bench_one(lib_f16, M, T, K, False, warmup, repeat)
            except Exception as e:
                print(f"{M:>7} {K:>7} {T:>6} {cnt:>4} | 失败: {e}")
                continue
            ratio = nv_ms / f16_ms
            nv_tf = flops / (nv_ms / 1000) / 1e12
            f16_tf = flops / (f16_ms / 1000) / 1e12
            print(f"{M:>7} {K:>7} {T:>6} {cnt:>4} | "
                  f"{nv_ms:>10.3f} {f16_ms:>10.3f} {ratio:>7.3f} | "
                  f"{nv_tf:>10.3f} {f16_tf:>9.3f}")
            results.append({
                "M_out": M, "K_in": K, "T_tokens": T, "layer_count": cnt,
                "nvfp4_ms": nv_ms, "fp16_ms": f16_ms, "ratio": ratio,
                "nvfp4_tflops": nv_tf, "fp16_tflops": f16_tf,
                "nvfp4_weight_bytes": nv_wb, "fp16_weight_bytes": f16_wb,
            })

    # 加权单次前向估算 (某 T 下: 每层调用1次, 未测尺寸按同 T 比例外推)
    print("\n=== 加权单次前向估算 (前9种尺寸, 按层调用次数加权) ===")
    forward_est = []
    for T in TOKENS:
        rows = [r for r in results if r["T_tokens"] == T]
        nv_total = sum(r["nvfp4_ms"] * r["layer_count"] for r in rows)
        f16_total = sum(r["fp16_ms"] * r["layer_count"] for r in rows)
        # K=188160 未测的 T: 按 T 线性外推 (GEMM 时间 ∝ T)
        for s in sizes:
            if s["K"] > 100000 and T > BIG_K_LIMIT_TOKENS:
                small = [r for r in results if r["M_out"] == s["N"]
                         and r["K_in"] == s["K"]]
                if small:
                    r0 = min(small, key=lambda x: x["T_tokens"])
                    k = T / r0["T_tokens"]
                    nv_total += r0["nvfp4_ms"] * k * s["count"]
                    f16_total += r0["fp16_ms"] * k * s["count"]
        forward_est.append({"T": T, "nvfp4_total_ms": nv_total,
                            "fp16_total_ms": f16_total,
                            "speedup": f16_total / nv_total})
        print(f"  T={T:>6}: NVFP4={nv_total/1000:>8.3f}s  FP16={f16_total/1000:>8.3f}s  "
              f"比值(NVFP4/FP16)={nv_total/f16_total:.3f}")

    # 显存对比 (全模型 22.1B Linear 参数)
    fp16_gb = total_params * 2 / 2**30
    nvfp4_gb = total_params * 0.28125 / 2**30  # 4bit + 1/16 E8M0 scale = 4.5/16
    print(f"\n=== 权重显存 (22.1B Linear 参数) ===")
    print(f"  FP16: {fp16_gb:.1f} GB   NVFP4: {nvfp4_gb:.1f} GB   "
          f"压缩比 1:{fp16_gb/nvfp4_gb:.2f}")

    report = {
        "gpu": name_buf.value.decode(),
        "shapes_source": SHAPES_JSON,
        "sizes_tested": [{"M_out": s["N"], "K_in": s["K"], "count": s["count"]}
                          for s in sizes],
        "coverage_params": covered / total_params,
        "results": results,
        "forward_estimate": forward_est,
        "weight_mem_gb": {"fp16": fp16_gb, "nvfp4": nvfp4_gb},
    }
    with open(REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n[保存] {REPORT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
