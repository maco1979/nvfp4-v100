# -*- coding: utf-8 -*-
"""Qwen3.8-27B 权重流式 NVFP4 量化 (阶段A2)
- safetensors mmap 逐张量读取 (32GB RAM 安全, 峰值 <2GB)
- 向量化 encode (语义对齐 nvfp4_codec.encode_fp16_to_nvfp4, 自检通过才放行)
- 输出: nvfp4_packed.bin + nvfp4_index.json + fp16_embed.npy + fp32_misc.npz
跳过: model.visual.* (纯文本), *mtp* (不用 speculative)
"""
import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "codec"))
sys.stdout.reconfigure(encoding="utf-8")

MODEL_DIR = os.environ.get("NVFP4_MODEL_DIR", r"F:\models\Qwen3.8-27B")
OUT_BIN = os.path.join(MODEL_DIR, "nvfp4_packed.bin")
OUT_IDX = os.path.join(MODEL_DIR, "nvfp4_index.json")
OUT_EMB = os.path.join(MODEL_DIR, "fp16_embed.npy")
OUT_MISC = os.path.join(MODEL_DIR, "fp32_misc.npz")

E2M1_MAX_ABS = 6.0
BLOCK = 16
BIAS = 127
ROW_CHUNK = 4096  # 分块行数, 控制 argmin 8x 中间内存


def encode_block_scales(max_abs_f64: np.ndarray) -> np.ndarray:
    """(M, K/16) float64 max_abs -> uint8 scale bytes. 语义=codec.compute_block_scale"""
    with np.errstate(divide="ignore", invalid="ignore"):
        scale_exp = np.ceil(np.log2(max_abs_f64 / E2M1_MAX_ABS))
    scale_exp = np.where(max_abs_f64 == 0.0, 0.0, scale_exp)
    scale_exp = np.clip(scale_exp, -127, 127)
    return (scale_exp + BIAS).astype(np.uint8), scale_exp  # bytes, exp(float64)


def e2m1_codes_rtn(abs_x: np.ndarray) -> np.ndarray:
    """RTN 最近邻 E2M1 编码, 平局取小码点 (与 argmin 语义一致)"""
    CP = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)
    EDGES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)
    a = np.clip(abs_x, 0.0, 6.0)
    # digitize: 平局(x==edge)取小码点 -> side 处理: x < edge ? left : right
    # np.searchsorted(EDGES, a, side='left'): a==edge 时返回该边索引 -> 码点=该索引 (取小) ✓
    return np.searchsorted(EDGES, a, side="left").astype(np.uint8)


def encode_rows(w_f32: np.ndarray):
    """(rows, K) float32 -> packed (rows, K/2) uint8, scale_bytes (rows, K/16)"""
    M, K = w_f32.shape
    blocks = w_f32.reshape(M, K // BLOCK, BLOCK)
    max_abs = np.abs(blocks).max(axis=-1).astype(np.float64)  # (M, K/16)
    sbytes, sexp = encode_block_scales(max_abs)
    scales_f = np.exp2(sexp)  # float64
    norm = (blocks / scales_f[..., None]).astype(np.float32)  # (M, K/16, 16)
    flat = norm.reshape(M, K)
    signs = (flat < 0).astype(np.uint8)
    signs[flat == 0] = 0
    codes = e2m1_codes_rtn(np.abs(flat))
    nib = (signs << 3) | codes
    packed = (nib[:, 0::2] & 0x0F) | ((nib[:, 1::2] & 0x0F) << 4)
    return packed, sbytes


def self_test():
    """与 nvfp4_codec 原版逐元素一致性自检"""
    import nvfp4_codec as ref
    rng = np.random.default_rng(7)
    w = (rng.standard_normal((64, 512)) * rng.choice([0.01, 1.0, 100.0], (64, 512))).astype(np.float16)
    w[0, :16] = 0.0  # 全零 block
    p_ref, s_ref = ref.encode_fp16_to_nvfp4(w)
    p_new, s_new = encode_rows(w.astype(np.float32))
    assert np.array_equal(s_ref, s_new), "scale 不一致!"
    assert np.array_equal(p_ref, p_new), "packed 不一致!"
    print("[self-test] 向量化 encode 与 codec 原版完全一致 ✓")


def main():
    self_test()
    shards = sorted(
        os.path.join(MODEL_DIR, f) for f in os.listdir(MODEL_DIR)
        if f.startswith("model-") and f.endswith(".safetensors"))
    assert len(shards) == 18, f"分片不齐: {len(shards)}/18"
    print(f"{len(shards)} 分片就绪")

    index, misc = {}, {}
    t0 = time.time()
    total_params = 0
    with open(OUT_BIN, "wb") as out:
        for si, shard in enumerate(shards):
            with safe_open(shard, framework="pt", device="cpu") as f:
                keys = sorted(f.keys())
                for name in keys:
                    t = f.get_tensor(name)  # mmap 按需
                    if name.startswith("model.visual.") or "mtp" in name.lower():
                        continue
                    if t.dim() == 2:
                        M, K = t.shape
                        w = t.to(torch.float16).numpy()
                        if "embed_tokens" in name:
                            np.save(OUT_EMB, w)
                            print(f"  [emb] {name} {M}x{K} -> fp16 npy", flush=True)
                            continue
                        # 分块量化
                        pk_parts, sb_parts = [], []
                        for r0 in range(0, M, ROW_CHUNK):
                            p, s = encode_rows(w[r0:r0 + ROW_CHUNK].astype(np.float32))
                            pk_parts.append(p)
                            sb_parts.append(s)
                        packed = np.concatenate(pk_parts)
                        scales = np.concatenate(sb_parts)
                        off_p = out.tell()
                        out.write(packed.tobytes())
                        off_s = out.tell()
                        out.write(scales.tobytes())
                        index[name] = {"M": M, "K": K,
                                       "packed_off": off_p, "packed_bytes": packed.nbytes,
                                       "scales_off": off_s, "scales_bytes": scales.nbytes,
                                       "fmt": "nvfp4_v2_embedded_sign"}
                        total_params += M * K
                    else:
                        misc[name] = t.to(torch.float32).numpy()
            print(f"[{si+1}/18] {os.path.basename(shard)} done, 累计 {total_params/1e9:.2f}B, "
                  f"{(time.time()-t0)/60:.1f}min", flush=True)

    np.savez(OUT_MISC, **misc)
    with open(OUT_IDX, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=1)
    gbytes = os.path.getsize(OUT_BIN) / 1e9
    print(f"\n量化完成: {total_params/1e9:.2f}B params -> {gbytes:.2f} GB "
          f"(有效密度 {gbytes*8/total_params:.3f} bit/param)")
    print(f"index: {len(index)} tensors | misc: {len(misc)} | 耗时 {(time.time()-t0)/60:.1f}min")


if __name__ == "__main__":
    main()
