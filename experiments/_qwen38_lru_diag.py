# -*- coding: utf-8 -*-
"""诊断: LRU serve P1 FAIL 根因隔离
==========================================================
ref1  = baseline P1 (graph 捕获+restore 后的首次 decode)
D1    = poison → 全量 prefill → refresh → set_pos → decode   (无快照)
D2    = poison → prefill前缀 → snap → prefill后缀 → refresh → set_pos → decode
对比各自与 ref1 的首个发散 token; CUDA_LAUNCH_BLOCKING=1 同步定位崩溃。
用法: python _qwen38_lru_diag.py
"""
import os

os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")
import sys
import time

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")
from _qwen38_infer import (build_model, ensure_kv_stage, snap_decode_state,
                           restore_decode_state, patch_fused_deltanet,
                           refresh_deltanet_state, GraphDecoder)
from _qwen38_p4a_lru import (P1_TEXT, SUFFIX, prefill, decode_n, snap_dn,
                             restore_dn, poison_cache)
from transformers import StaticCache

CHUNK = 512
NGEN = 31
MAXLEN = 3072


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i, x, y
    if len(a) != len(b):
        return min(len(a), len(b)), "LEN", f"{len(a)} vs {len(b)}"
    return None


def main():
    print("[0] 建模 ...", flush=True)
    tok, model = build_model()
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu
    hid = emb_w.shape[1]
    eos = tok.eos_token_id

    ids_p1 = tok(P1_TEXT, return_tensors="pt").input_ids
    Lp = (ids_p1.shape[1] // CHUNK) * CHUNK
    p = ids_p1[:, :Lp].cuda()
    t = tok.apply_chat_template([{"role": "user", "content": SUFFIX}],
                                tokenize=False, add_generation_prompt=True,
                                enable_thinking=False)
    suf = tok(t, return_tensors="pt").input_ids.cuda()
    f = torch.cat([p, suf], dim=1)
    L = f.shape[1]
    print(f"[数据] 前缀 {Lp} + 后缀 {suf.shape[1]} = {L} tok", flush=True)

    # === ref1: 标准 baseline 流 (与 _qwen38_p4a_lru.py baseline k=0 相同) ===
    poison_cache(cache)
    first = prefill(model, cache, f, 0, L)
    mods = patch_fused_deltanet(model, cache)
    vf_snap = [(m._vf_conv.clone().cpu(), m._vf_S.clone().cpu()) for m in mods]
    kv_snap = snap_decode_state(model, cache, L)
    dec = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap, L)
    for m, (c, s) in zip(mods, vf_snap):
        m._vf_conv.copy_(c)
        m._vf_S.copy_(s)
    dec.reset_pos()
    ref1 = decode_n(dec, first, emb_w, eos)
    print(f"[ref1] first={first} len={len(ref1)} "
          f"head={ref1[:8]}", flush=True)

    # === D1: 无快照第二次 decode 会话 ===
    poison_cache(cache)
    first1 = prefill(model, cache, f, 0, L)
    refresh_deltanet_state(model, cache)
    dec.set_pos(L)
    d1 = decode_n(dec, first1, emb_w, eos)
    fd = first_diff(ref1, d1)
    print(f"[D1 无快照] first={first1} 一致={'PASS' if fd is None else 'FAIL'} "
          f"{'' if fd is None else f'发散@{fd}'}", flush=True)

    # === D2: 前缀+快照+后缀 (serve MISS 路径) ===
    poison_cache(cache)
    prefill(model, cache, p, 0, Lp)
    kv, dn = snap_decode_state(model, cache, Lp), snap_dn(model, cache)
    first2 = prefill(model, cache, f, Lp, L)
    refresh_deltanet_state(model, cache)
    dec.set_pos(L)
    d2 = decode_n(dec, first2, emb_w, eos)
    fd = first_diff(ref1, d2)
    print(f"[D2 快照分割] first={first2} 一致={'PASS' if fd is None else 'FAIL'} "
          f"{'' if fd is None else f'发散@{fd}'}", flush=True)

    # === D2b: 快照分割但不存缓存 (排除 put 副作用——本来也无) + 验证恢复路径 ===
    # (D2 已含, 此处直接测 restore: poison → restore 前缀 → prefill 后缀)
    poison_cache(cache)
    restore_decode_state(model, cache, kv, Lp)
    restore_dn(cache, dn)
    first3 = prefill(model, cache, f, Lp, L)
    refresh_deltanet_state(model, cache)
    dec.set_pos(L)
    d3 = decode_n(dec, first3, emb_w, eos)
    fd = first_diff(ref1, d3)
    print(f"[D3 restore路径] first={first3} 一致={'PASS' if fd is None else 'FAIL'} "
          f"{'' if fd is None else f'发散@{fd}'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
