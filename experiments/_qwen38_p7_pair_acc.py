# -*- coding: utf-8 -*-
"""P7-2 验收: 成对 GEMV 融合 (63 token 逐位 + tok/s, P6 基线 33.9)
================================================================================
流程 (同进程):
  Run1: baseline GraphDecoder 63 token (P5-A+P6 全 patch, 无 P7)
  Run2: patch_pair_gemv → 重 prefill + 重 capture → 63 token 逐位对比 + tok/s
用法: python _qwen38_p7_pair_acc.py
"""
import sys
import time

import torch

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")
from _qwen38_infer import (build_model, ensure_kv_stage, snap_decode_state,
                           restore_decode_state, patch_fused_deltanet,
                           patch_attn_decode, patch_rmsnorm,
                           refresh_deltanet_state, GraphDecoder)
from _qwen38_p4a_lru import P1_TEXT, SUFFIX, prefill, poison_cache
from _qwen38_p7_pair import patch_pair_gemv
from transformers import StaticCache

MAXLEN = 3072
NGEN = 63
PROMPT = P1_TEXT + SUFFIX


def decode_n(dec, first, emb_w, eos):
    gen, cur = [first], first
    with torch.no_grad():
        for _ in range(NGEN):
            cur = dec.step(cur, emb_w)
            gen.append(cur)
            if eos is not None and cur == eos:
                break
    torch.cuda.synchronize()
    return gen


def bench(dec, first, emb_w):
    """graph replay 计时: 与既有验收同口径 (63 tok, 去 warmup)"""
    dec.reset_pos()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    n = 0
    cur = first
    with torch.no_grad():
        for _ in range(NGEN):
            cur = dec.step(cur, emb_w)
            n += 1
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return n / dt, dt * 1e3 / n


def main():
    print("[0] 建骨架 + NVFP4 权重上传 ...", flush=True)
    tok, model = build_model()
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu
    hid = emb_w.shape[1]
    eos = tok.eos_token_id

    ids = tok(PROMPT, return_tensors="pt").input_ids
    L = ids.shape[1]
    ids = ids.cuda()
    print(f"[数据] prompt {L} tok", flush=True)

    # === Run1: baseline (P5-A + P6, 无 P7) ===
    poison_cache(cache)
    first = prefill(model, cache, ids, 0, L)
    patch_fused_deltanet(model, cache)
    patch_rmsnorm(model)
    patch_attn_decode(model, cache)
    refresh_deltanet_state(model, cache)
    kv_snap = snap_decode_state(model, cache, L)
    dec0 = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap, L)
    refresh_deltanet_state(model, cache)
    dec0.reset_pos()
    ref = decode_n(dec0, first, emb_w, eos)
    print(f"[Run1 baseline] first={first} len={len(ref)} head={ref[:8]}",
          flush=True)
    del dec0, kv_snap
    torch.cuda.empty_cache()

    # === Run2: P7-2 成对融合 → 重 prefill + 重 capture ===
    patch_pair_gemv(model)
    poison_cache(cache)
    first2 = prefill(model, cache, ids, 0, L)
    refresh_deltanet_state(model, cache)
    kv_snap2 = snap_decode_state(model, cache, L)
    dec1 = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap2, L)
    refresh_deltanet_state(model, cache)
    dec1.reset_pos()
    out = decode_n(dec1, first2, emb_w, eos)
    same = len(out) == len(ref) and out == ref
    print(f"[Run2 P7-2] first={first2} len={len(out)} head={out[:8]}", flush=True)
    print(f"[验收] 63 token 逐位: {'PASS' if same else 'FAIL'} "
          f"({len(out)}/{len(ref)})", flush=True)
    if not same:
        for i, (a, b) in enumerate(zip(ref, out)):
            if a != b:
                print(f"  首个分歧 @{i}: ref={a} out={b}")
                break

    tps, ms = bench(dec1, first2, emb_w)
    print(f"[性能] {tps:.1f} tok/s ({ms:.2f} ms/步) | P6 基线 33.9 "
          f"({(tps-33.9)/33.9*100:+.1f}%)", flush=True)
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())
