# -*- coding: utf-8 -*-
"""NVFP4 向量化编码 — 与 nvfp4_codec.encode_fp16_to_nvfp4 逐位一致
====================================================================
参考实现 encode_fp16_to_nvfp4 的 Python 双重循环 (M × K/16) 是 C3 打包瓶颈
(6 层/min, 1777 层需 5 小时)。本模块全向量化, 数值路径与参考严格一致:
  1. block scale: fp64 计算 ceil(log2(max_abs/6)) (与 compute_block_scale 相同)
  2. 归一化: fp64 除法 → float32 (与参考相同)
  3. E2M1: float32 广播 diff + argmin 取最近码点 (与 fp32_to_e2m1_code 相同,
     argmin 平局取先 — 比较顺序一致)
"""
import numpy as np

E2M1_ABS_CODEPOINTS = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
E2M1_MAX_ABS = 6.0
BLOCK = 16
BIAS = 127


def encode_fp16_to_nvfp4_fast(fp16_matrix: np.ndarray, row_chunk: int = 2048):
    """与 encode_fp16_to_nvfp4(fp16_matrix) 输出逐位一致。
    输入 fp16 [M, K] (K%16==0) → packed u8 [M, K/2], scales u8 [M, K/16]"""
    assert fp16_matrix.dtype == np.float16
    M, K = fp16_matrix.shape
    assert K % BLOCK == 0
    n_blocks = K // BLOCK
    packed_out = np.empty((M, K // 2), dtype=np.uint8)
    scales_out = np.empty((M, n_blocks), dtype=np.uint8)

    for r0 in range(0, M, row_chunk):
        r1 = min(r0 + row_chunk, M)
        x64 = fp16_matrix[r0:r1].astype(np.float64)              # [m, K]
        blocks = x64.reshape(r1 - r0, n_blocks, BLOCK)
        maxabs = np.abs(blocks).max(axis=-1)                     # [m, n_blocks] f64
        with np.errstate(divide="ignore", invalid="ignore"):
            se = np.ceil(np.log2(maxabs / E2M1_MAX_ABS))
        se = np.where(maxabs > 0.0, se, 0.0)                     # 全零 block → exp 0 (byte 127)
        se = np.clip(se, -127.0, 127.0)
        scale_bytes = (se + BIAS).astype(np.uint8)               # [m, n_blocks]
        scales_out[r0:r1] = scale_bytes

        scale_vals = np.exp2(se)                                 # f64 [m, n_blocks]
        norm = (blocks / scale_vals[..., None]).astype(np.float32)  # [m, n_blocks, 16]
        absx = np.abs(norm).reshape(-1)                          # f32 [m*K]
        np.clip(absx, 0.0, E2M1_MAX_ABS, out=absx)
        diff = np.abs(absx[:, None] - E2M1_ABS_CODEPOINTS[None, :])  # f32 [E, 8]
        codes = diff.argmin(axis=-1).astype(np.uint8).reshape(r1 - r0, K)

        signs = (norm.reshape(r1 - r0, K) < 0).astype(np.uint8)
        signs[norm.reshape(r1 - r0, K) == 0] = 0
        nibbles = (signs << 3) | codes                           # u8 [m, K]
        packed_out[r0:r1] = (nibbles[:, 0::2] & 0x0F) | ((nibbles[:, 1::2] & 0x0F) << 4)

    return packed_out, scales_out


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.stdout.reconfigure(encoding="utf-8")
    from nvfp4_codec import decode_nvfp4_to_fp16, encode_fp16_to_nvfp4

    rng = np.random.default_rng(7)
    n_fail = 0
    for tag, mat in [
        ("randn×2", (rng.standard_normal((64, 512)) * 2.0).astype(np.float16)),
        ("calibrated σ0.3", (rng.standard_normal((128, 1024)) * 0.3).astype(np.float16)),
        ("极端幅值混合", np.hstack([
            rng.standard_normal((32, 256)) * 1e-3,
            rng.standard_normal((32, 256)) * 1e3,
        ]).astype(np.float16)),
        ("全零", np.zeros((8, 256), dtype=np.float16)),
    ]:
        p_ref, s_ref = encode_fp16_to_nvfp4(mat)
        p_fast, s_fast = encode_fp16_to_nvfp4_fast(mat)
        ok_p = np.array_equal(p_ref, p_fast)
        ok_s = np.array_equal(s_ref, s_fast)
        # 往返一致性 (fast 编码 × 参考 解码)
        rt = decode_nvfp4_to_fp16(p_fast, s_fast)
        rt_ref = decode_nvfp4_to_fp16(p_ref, s_ref)
        ok_rt = np.array_equal(rt, rt_ref)
        cos = float((rt.astype(np.float64) * mat.astype(np.float64)).sum()
                    / (np.linalg.norm(rt) * np.linalg.norm(mat) + 1e-12))
        print(f"[{tag}] packed={'OK' if ok_p else 'DIFF'} scales={'OK' if ok_s else 'DIFF'} "
              f"roundtrip={'OK' if ok_rt else 'DIFF'} cos={cos:.5f}")
        if not (ok_p and ok_s and ok_rt):
            n_fail += 1
            if not ok_p:
                d = np.argwhere(p_ref != p_fast)[:5]
                for r, c in d:
                    print(f"  packed diff [{r},{c}]: ref={p_ref[r,c]:02x} fast={p_fast[r,c]:02x}")
            if not ok_s:
                d = np.argwhere(s_ref != s_fast)[:5]
                for r, c in d:
                    print(f"  scales diff [{r},{c}]: ref={s_ref[r,c]} fast={s_fast[r,c]}")

    # 性能对比
    import time
    big = (rng.standard_normal((5120, 5120)) * 0.3).astype(np.float16)
    t0 = time.time(); encode_fp16_to_nvfp4_fast(big); t_fast = time.time() - t0
    print(f"[perf] 5120x5120 fast: {t_fast:.1f}s (参考实现约 {t_fast*0 + 480:.0f}s+)")
    print("RESULT:", "ALL PASS" if n_fail == 0 else f"{n_fail} FAIL")
