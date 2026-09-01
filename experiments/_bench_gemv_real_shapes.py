# -*- coding: utf-8 -*-
"""P7-1: 真实 shape 隔离 GEMV bench —— 裁决 663 vs 792 GB/s 之争
================================================================================
背景: 隔离基线 792GB/s 只测过 4 个大 shape (12288×5120 等), 而模型每步实际
launch 497 个 GEMV、9 种尺寸, 其中 96 个 48×5120 微型 (grid 仅 48 block,
无法占满 80 SM) + 32 个 1024×5120 小型。in-graph 平均 663GB/s 的差距可能
来自尺寸混合效应而非调度干扰。
方法: 对 9 种真实 shape × 真实次数, 用同一 DLL(v8) 隔离 bench:
  - 大 shape (>L2): 单 buffer 重复 launch (无 L2 复用, 诚实)
  - 小 shape (≤~3MB): 分配 N 个独立 buffer 轮转 (总字节 >6MB L2, 每次冷读)
输出: 每 shape 单核时长 + 带宽 → 加权预测每步 GEMV 总时长, 对比 in-graph 21.73ms。
用法: python _bench_gemv_real_shapes.py
"""
import ctypes
import os
import sys
import time

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

sys.stdout.reconfigure(encoding="utf-8")
DLL = os.path.join(_ROOT, "kernels", "nvfp4_gemv.dll")

# (M, K, 每步实例数) —— 来自 nvfp4_index.json 497 个 QLinear 的真实分布
SHAPES = [
    (248320, 5120, 1),    # lm_head
    (5120, 17408, 64),    # mlp down
    (17408, 5120, 128),   # mlp up/gate
    (12288, 5120, 16),    # attn qkvz 大层
    (10240, 5120, 48),    # in_proj_qkz
    (6144, 5120, 48),     # in_proj_ba
    (5120, 6144, 64),     # o_proj 等
    (1024, 5120, 32),     # conv/小投影
    (48, 5120, 96),       # linear_attn in_proj_a/b
]
L2_BYTES = 6 * 1024 * 1024   # V100 6MB
REPEAT = 30                  # 每 shape 轮数


def main():
    lib = ctypes.CDLL(DLL)
    lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib.gpu_malloc.restype = ctypes.c_void_p
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_h2d.restype = ctypes.c_int
    lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_d2h.restype = ctypes.c_int
    L = lib.launch_nvfp4_gemv_v8
    L.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    L.restype = ctypes.c_int

    sink = np.empty(248320, np.float16)
    pred_ms = 0.0
    bytes_tot = 0
    print(f"{'shape':<18} {'次/步':>5} {'μs/核':>8} {'GB/s':>6} {'预测ms/步':>9}")
    print("-" * 56)
    for M, K, cnt in SHAPES:
        rng = np.random.default_rng(M + K)
        # 带宽测试与数据内容无关: packed 用随机字节, scales 用随机 fp16
        pb = rng.integers(0, 255, M * K // 2, dtype=np.uint8).tobytes()
        sb = (rng.standard_normal(M * K // 16) * 0.1).astype(np.float16).tobytes()
        ab = (rng.standard_normal(K) * 0.5).astype(np.float16).tobytes()
        n_bytes = M * K // 2 + M * K // 16 + K * 2 + M * 2
        # 小权重: 分配多块独立 buffer 轮转, 总量>L2 → 每次冷读
        n_buf = max(1, (L2_BYTES // max(len(pb), 1)) + 1)
        if M * K <= 48 * 5120 * 2:            # 微型 shape 全量实例
            n_buf = cnt
        bufs = []
        for i in range(n_buf):
            d_p, d_s = lib.gpu_malloc(len(pb)), lib.gpu_malloc(len(sb))
            assert d_p and d_s
            assert lib.gpu_memcpy_h2d(d_p, pb, len(pb)) == 0
            assert lib.gpu_memcpy_h2d(d_s, sb, len(sb)) == 0
            bufs.append((d_p, d_s))
        d_a, d_o = lib.gpu_malloc(len(ab)), lib.gpu_malloc(M * 2)
        assert d_a and d_o
        assert lib.gpu_memcpy_h2d(d_a, ab, len(ab)) == 0
        print(f"  [probe] {M}x{K} alloc+h2d done, warm launch ...", flush=True)
        L(bufs[0][0], bufs[0][1], d_a, d_o, M, K, ctypes.c_void_p(0))  # warm
        lib.gpu_memcpy_d2h(sink[: M].ctypes.data, d_o, M * 2)
        print(f"  [probe] warm ok", flush=True)
        best = 1e9
        host_out = np.empty(M, np.float16)          # 预分配: 防 ctypes 临时数组被 GC
        for _ in range(3):
            t1 = time.perf_counter()
            for r in range(REPEAT):
                dp, ds = bufs[r % n_buf]
                L(dp, ds, d_a, d_o, M, K, ctypes.c_void_p(0))
            lib.gpu_memcpy_d2h(host_out.ctypes.data, d_o, M * 2)
            best = min(best, (time.perf_counter() - t1) / REPEAT)
        step_ms = best * 1e3 * cnt
        bw = n_bytes / best / 1e9
        pred_ms += step_ms
        bytes_tot += n_bytes * cnt
        print(f"{M}x{K:<10} {cnt:>5} {best*1e6:>8.1f} {bw:>6.0f} {step_ms:>9.2f}")
        for dp, ds in bufs:
            lib.gpu_free(dp)
            lib.gpu_free(ds)
        lib.gpu_free(d_a)
        lib.gpu_free(d_o)

    print("-" * 56)
    print(f"预测每步 GEMV 总时长 (隔离加权): {pred_ms:.2f} ms")
    print(f"每步权重字节: {bytes_tot/1e9:.2f} GB → 加权带宽 "
          f"{bytes_tot/1e6/pred_ms:.0f} GB/s")
    print(f"[裁决] in-graph 实测 21.73 ms / 663 GB/s")
    print(f"  差值 = {pred_ms - 21.73:+.2f} ms "
          f"(负值=隔离更快, 干扰/调度损失真实存在; 正值≈0=尺寸混合即极限)")


if __name__ == "__main__":
    sys.exit(main())
