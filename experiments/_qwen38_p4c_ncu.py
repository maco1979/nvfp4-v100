# -*- coding: utf-8 -*-
"""S1-NCU: 剖析 decode graph 内部内核构成 (21.6ms 非 GEMV 归因)
=====================================================================
方法: ncu --graph-profiling=node 挂本进程, 采集 graph replay 中
      每个内核节点的 gpu__time_duration (ns)。本脚本只负责:
      建模 → prefill → 捕获 graph → NCU_PROFILE_ONLY 环境变量
      控制下 replay 少量步 (每内核只采 1 次, --launch-count 限制)。
用法: 先单独跑一次做 warmup 无关 — NCU 全程包裹即可:
  ncu --graph-profiling=node --metrics gpu__time_duration.sum \
      --csv --page raw -o 无需 -f python _qwen38_p4c_ncu.py
"""
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")
from _qwen38_infer import (build_model, ensure_kv_stage, snap_decode_state,
                           restore_decode_state, patch_fused_deltanet,
                           patch_attn_decode, patch_rmsnorm,
                           refresh_deltanet_state, GraphDecoder)
from transformers import StaticCache

CHUNK = 512
NSTEP = int(os.environ.get("NCU_STEPS", "3"))
MAXLEN = 3072
PROMPT = "用一句话解释什么是块缩放浮点量化。"


def main():
    print("[1] 建骨架 + NVFP4 权重上传 ...", flush=True)
    tok, model = build_model()
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu
    hid = emb_w.shape[1]

    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    L = ids.shape[1]
    last = None
    with torch.no_grad():
        for i in range(0, L, CHUNK):
            j = min(i + CHUNK, L)
            out = model(ids[:, i:j], past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(i, j, device="cuda"),
                        logits_to_keep=1)
            if j == L:
                last = out.logits[:, -1, :]
    torch.cuda.synchronize()
    cur = int(last.argmax(dim=-1).item())

    mods = patch_fused_deltanet(model, cache)
    patch_rmsnorm(model)                     # P5-A: norm 融合
    patch_attn_decode(model, cache)          # P6: attn v3 kernel
    vf_snap = [(m._vf_conv.clone().cpu(), m._vf_S.clone().cpu()) for m in mods]
    kv_snap = snap_decode_state(model, cache, L)
    dec = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap, L)
    for m, (c, s) in zip(mods, vf_snap):
        m._vf_conv.copy_(c)
        m._vf_S.copy_(s)
    dec.reset_pos()
    print(f"[2] graph 捕获完成, replay {NSTEP} 步供 NCU 采集 ...", flush=True)

    # NCU 剖析窗口: 仅这几步 replay 中的内核会被计数
    torch.cuda.synchronize()
    for _ in range(NSTEP):
        cur = dec.step(cur, emb_w)
    torch.cuda.synchronize()
    print(f"[3] 完成, 末 token = {cur}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
