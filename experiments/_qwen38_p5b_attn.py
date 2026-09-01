# -*- coding: utf-8 -*-
"""P5-A: elementwise 融合验收 (attn gate 融合 + RMSNorm 全站融合)
==========================================================
- baseline: sdpa + R2 + torch 原生 norm 链, GraphDecoder 63 tok
- patch 组: patch_attn_decode v2 (gate·sigmoid 并入 kernel)
           + patch_rmsnorm (161 处 × 10 内核 → 1 内核)
- 验收: 63 token 逐位一致 + tok/s (P5-B 基线 28.8)
用法: python _qwen38_p5b_attn.py
"""
import ctypes
import os
import sys
import time

import torch

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
from transformers import StaticCache

MAXLEN = 3072
CHUNK = 512
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

    # === Run1: baseline (sdpa + R2 + torch norm, graph decode) ===
    poison_cache(cache)
    first = prefill(model, cache, ids, 0, L)
    mods_dn = patch_fused_deltanet(model, cache)
    refresh_deltanet_state(model, cache)
    kv_snap = snap_decode_state(model, cache, L)
    dec0 = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap, L)
    refresh_deltanet_state(model, cache)
    dec0.reset_pos()
    ref1 = decode_n(dec0, first, emb_w, eos)
    print(f"[Run1 baseline] first={first} len={len(ref1)} head={ref1[:8]}",
          flush=True)
    del dec0, kv_snap
    torch.cuda.empty_cache()

    # === Run2a: RMSNorm kernel 单元测试 (vs torch 原生) ===
    norms = patch_rmsnorm(model)
    final_norm = model.model.norm
    n_h = final_norm.weight.shape[0]
    torch.manual_seed(0)
    x_a = (torch.randn(1, 1, n_h, device="cuda") * 2).half()
    y_k = final_norm(x_a)
    y_t = final_norm._vf_norm_orig(x_a)
    da = (y_k.float() - y_t.float()).abs().max().item()
    # chunk view 用例 (q_norm 布局: 行间 gap=512, n=256)
    q_norm = next(l.self_attn.q_norm for l in model.model.layers
                  if l.block_type == "full_attention")
    qg = (torch.randn(1, 1, 24, 512, device="cuda") * 2).half()
    x_b = torch.chunk(qg, 2, dim=-1)[0].view(1, 1, 24, 256)
    assert x_b.stride(-2) == 512 and x_b.shape[-1] == 256
    y_kb = q_norm(x_b)
    y_tb = q_norm._vf_norm_orig(x_b)
    db = (y_kb.float() - y_tb.float()).abs().max().item()
    ne = (y_k != y_t).sum().item()
    neb = (y_kb != y_tb).sum().item()
    print(f"[Run2a 单元] {n_h}维: maxdiff={da:.2e} 非零={ne}/{n_h} | "
          f"256维chunk: maxdiff={db:.2e} 非零={neb}/6144", flush=True)

    # === Run2: patch attn v2 + rmsnorm + graph decode ===
    mods_attn = patch_attn_decode(model, cache)
    poison_cache(cache)
    first2 = prefill(model, cache, ids, 0, L)     # T>1 走原路径
    refresh_deltanet_state(model, cache)
    kv_snap = snap_decode_state(model, cache, L)
    dec = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap, L)
    refresh_deltanet_state(model, cache)
    dec.reset_pos()
    d = decode_n(dec, first2, emb_w, eos)
    ok = d == ref1
    print(f"[Run2 kernel]   first={first2} len={len(d)} head={d[:8]} "
          f"{'PASS' if ok else 'FAIL'}", flush=True)

    # === Run3: 性能 (graph replay 512 步均值) ===
    t0 = time.time()
    with torch.no_grad():
        for _ in range(512):
            dec.step(109015, emb_w)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tps = 512 / dt
    print(f"[Run3 perf] {tps:.1f} tok/s ({dt*1000/512:.2f} ms/step)", flush=True)

    print(f"\n===== P5-A 验收 =====")
    print(f"norm 单元: 4096维 maxdiff={da:.2e} / 256维 maxdiff={db:.2e} (fp16 舍入级)")
    print(f"token 逐位一致: {'PASS' if ok else 'FAIL'}")
    print(f"decode 速度: {tps:.1f} tok/s (P5-B 28.8, baseline 24.8)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
