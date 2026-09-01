# -*- coding: utf-8 -*-
"""Qwen3.8-27B 真实尺寸回放测速 (阶段B, 基于 nvfp4_ltx_replay_bench 框架)
尺寸表: 从 safetensors header 实测推导 (2026-08-16)
场景: decode T=1 / 小batch T=64 / prefill T=512,2048
"""
import ctypes
import json
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
V5_DLL = os.path.join(_ROOT, "kernels", "nvfp4_cuda_v5.dll")
F16_DLL = os.path.join(_ROOT, "kernels", "nvfp4_baseline_f16.dll")
OUT_JSON = os.path.join(_ROOT, "experiments", "data", "qwen38_replay_bench.json")

# (name, M=out_features, K=in_features, count)  —— kernel: weight[M,K]@act[K,T]
# 来源: 18 分片 safetensors header 扫描 + 64层外推
SHAPES = [
    ("mlp.gate_proj",   17408, 5120, 64),
    ("mlp.up_proj",     17408, 5120, 64),
    ("mlp.down_proj",    5120, 17408, 64),
    ("la.in_proj_qkv",  10240, 5120, 48),
    ("la.in_proj_z",     6144, 5120, 48),
    ("la.out_proj",      5120, 6144, 48),
    ("sa.q_proj",       12288, 5120, 16),
    ("sa.o_proj",         5120, 6144, 16),
    ("lm_head",        248320, 5120, 1),
]
# 忽略的小投影 (性能影响 <0.5%): sa.k/v (1024,5120)×16, la.a/b (48,5120)×48×2
TOKENS = [1, 64, 512, 2048]


def load_libs():
    lib = ctypes.CDLL(V5_DLL)
    lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib.gpu_malloc.restype = ctypes.c_void_p
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_h2d.restype = ctypes.c_int
    lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_d2h.restype = ctypes.c_int
    lib.launch_nvfp4_gemmtc_v5.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_nvfp4_gemmtc_v5.restype = ctypes.c_int

    lib16 = ctypes.CDLL(F16_DLL)
    lib16.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib16.gpu_malloc.restype = ctypes.c_void_p
    lib16.gpu_free.argtypes = [ctypes.c_void_p]
    lib16.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib16.gpu_memcpy_h2d.restype = ctypes.c_int
    lib16.launch_fp16_gemm_2stage.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib16.launch_fp16_gemm_2stage.restype = ctypes.c_int
    return lib, lib16


def bench(lib, M, T, K, nvfp4, warmup=5, repeat=20):
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((M, K), dtype=np.float32) * 0.02).astype(np.float16)
    a = (rng.standard_normal((K, T), dtype=np.float32) * 0.5).astype(np.float16)
    d_o = lib.gpu_malloc(M * T * 2)
    if nvfp4:
        from _qwen38_nvfp4_pack import encode_rows
        pk_parts, sb_parts = [], []
        for r0 in range(0, M, 2048):
            p, s = encode_rows(w[r0:r0 + 2048].astype(np.float32))
            pk_parts.append(p)
            sb_parts.append(s)
        pb = np.concatenate(pk_parts).tobytes()
        sb = np.concatenate(sb_parts).tobytes()
        ab = a.tobytes()
        d_p, d_s, d_a = lib.gpu_malloc(len(pb)), lib.gpu_malloc(len(sb)), lib.gpu_malloc(len(ab))
        lib.gpu_memcpy_h2d(d_p, pb, len(pb))
        lib.gpu_memcpy_h2d(d_s, sb, len(sb))
        lib.gpu_memcpy_h2d(d_a, ab, len(ab))
        launch, args, keep = lib.launch_nvfp4_gemmtc_v5, (d_p, d_s, d_a, d_o), (d_p, d_s, d_a)
        wbytes = len(pb) + len(sb)
    else:
        wb, ab = w.tobytes(), a.tobytes()
        d_w, d_a = lib.gpu_malloc(len(wb)), lib.gpu_malloc(len(ab))
        lib.gpu_memcpy_h2d(d_w, wb, len(wb))
        lib.gpu_memcpy_h2d(d_a, ab, len(ab))
        launch, args, keep = lib.launch_fp16_gemm_2stage, (d_w, d_a, d_o), (d_w, d_a)
        wbytes = len(wb)
    for _ in range(warmup):
        launch(*args, M, T, K)
    best = 1e9
    for _ in range(repeat):
        t0 = time.perf_counter()
        launch(*args, M, T, K)
        best = min(best, (time.perf_counter() - t0) * 1000)
    for p in keep + (d_o,):
        lib.gpu_free(p)
    return best, wbytes


def main():
    lib, lib16 = load_libs()
    print(f"{'shape':<18} {'M':>7} {'K':>6} {'T':>5} | {'nvfp4 ms':>9} {'f16 ms':>9} | "
          f"{'TF(nv)':>7} {'GB/s(nv)':>9} | {'spdup':>6}")
    results = []
    for name, M, K, cnt in SHAPES:
        for T in TOKENS:
            ms_n, wb = bench(lib, M, T, K, True)
            ms_f, _ = bench(lib16, M, T, K, False)
            tf = 2 * M * T * K / (ms_n / 1000) / 1e12
            bws = wb / (ms_n / 1000) / 1e9  # 权重带宽 (decode 关键指标)
            row = dict(name=name, M=M, K=K, T=T, cnt=cnt,
                       nvfp4_ms=round(ms_n, 4), f16_ms=round(ms_f, 4),
                       nvfp4_tf=round(tf, 2), nvfp4_wbw_gbs=round(bws, 1),
                       speedup=round(ms_f / ms_n, 2))
            results.append(row)
            print(f"{name:<18} {M:>7} {K:>6} {T:>5} | {ms_n:>9.3f} {ms_f:>9.3f} | "
                  f"{tf:>7.2f} {bws:>9.1f} | {ms_f/ms_n:>6.2f}")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1)
    # 加权 decode (T=1) 每 token 时间: 每层 kernel 时间 × count 求和
    t1 = sum(r["nvfp4_ms"] * r["cnt"] for r in results if r["T"] == 1)
    tp = sum(r["nvfp4_ms"] * r["cnt"] for r in results if r["T"] == 2048)
    print(f"\n[T=1 decode]  Linear 总耗时 {t1:.2f} ms/token -> 上限 {1000/t1:.1f} tok/s (纯Linear)")
    print(f"[T=2048 prefill] Linear 总耗时 {tp:.0f} ms -> {2048/(tp/1000):.1f} tok/s prefill")


if __name__ == "__main__":
    main()
