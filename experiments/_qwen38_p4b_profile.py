# -*- coding: utf-8 -*-
"""P4b: step 级 profile — 31.2ms/step 中非 GEMV 残余环节定位
=====================================================================
已知: v8 GEMV ~27ms (792GB/s)。本脚本分解 graph 外环节:
  CPU gather (emb_w[tok] → pinned) / H2D (pinned → static_embeds) /
  pos4.fill / replay 启动 (CPU 侧) / .item() 同步,
并加 CUDA event 测 GPU 侧 graph replay 真实时长。
用法: python _qwen38_p4b_profile.py
"""
import sys
import time

import torch

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")
from _qwen38_infer import (build_model, ensure_kv_stage, snap_decode_state,
                           restore_decode_state, patch_fused_deltanet,
                           GraphDecoder)
from transformers import StaticCache

CHUNK = 512
NSTEP = 512
MAXLEN = 3072
PROMPT = "用一句话解释什么是块缩放浮点量化。"


def main():
    print("[0] 建骨架 + NVFP4 权重上传 ...", flush=True)
    tok, model = build_model()
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu
    hid = emb_w.shape[1]

    ids = tok(PROMPT, return_tensors="pt").input_ids.cuda()
    L = ids.shape[1]
    last_logits = None
    with torch.no_grad():
        for i in range(0, L, CHUNK):
            j = min(i + CHUNK, L)
            out = model(ids[:, i:j], past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(i, j, device="cuda"),
                        logits_to_keep=1)
            if j == L:
                last_logits = out.logits[:, -1, :]
    torch.cuda.synchronize()
    cur = int(last_logits.argmax(dim=-1).item())

    mods = patch_fused_deltanet(model, cache)
    vf_snap = [(m._vf_conv.clone().cpu(), m._vf_S.clone().cpu()) for m in mods]
    kv_snap = snap_decode_state(model, cache, L)
    dec = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap, L)
    for m, (c, s) in zip(mods, vf_snap):
        m._vf_conv.copy_(c)
        m._vf_S.copy_(s)
    dec.reset_pos()

    dec.set_pos(L)

    # === 计时分解: 手动复刻 step() 逐段计时 ===
    pin = dec.pin
    se = dec.static_embeds
    pos4 = dec.pos4
    graph = dec.graph
    nxt = dec.static_next

    acc = {"gather": 0.0, "h2d": 0.0, "fill": 0.0, "replay_cpu": 0.0,
           "item": 0.0}
    ev0 = torch.cuda.Event(enable_timing=True)
    ev1 = torch.cuda.Event(enable_timing=True)
    gpu_replay_ms = 0.0

    # warmup 3 步
    for _ in range(3):
        pin.copy_(emb_w[cur]); se.copy_(pin.view(1, 1, -1), non_blocking=True)
        pos4.fill_(dec.rope_pos); graph.replay(); dec.rope_pos += 1
        cur = int(nxt.item())
    torch.cuda.synchronize()

    t_all0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(NSTEP):
            t0 = time.perf_counter()
            pin.copy_(emb_w[cur])
            t1 = time.perf_counter()
            se.copy_(pin.view(1, 1, -1), non_blocking=True)
            t2 = time.perf_counter()
            pos4.fill_(dec.rope_pos)
            t3 = time.perf_counter()
            ev0.record()
            graph.replay()
            ev1.record()
            t4 = time.perf_counter()
            dec.rope_pos += 1
            cur = int(nxt.item())
            t5 = time.perf_counter()
            torch.cuda.synchronize()
            acc["gather"] += t1 - t0
            acc["h2d"] += t2 - t1
            acc["fill"] += t3 - t2
            acc["replay_cpu"] += t4 - t3
            acc["item"] += t5 - t4
            gpu_replay_ms += ev0.elapsed_time(ev1)
    t_all = time.perf_counter() - t_all0

    step_ms = t_all / NSTEP * 1000
    print(f"\n===== P4b: {NSTEP} step 分解 (每 step {step_ms:.2f}ms, "
          f"{NSTEP/t_all:.2f} tok/s) =====", flush=True)
    print(f"  CPU gather (emb_w[tok]→pin)   : {acc['gather']/NSTEP*1000:7.3f} ms", flush=True)
    print(f"  H2D (pin→static_embeds, async): {acc['h2d']/NSTEP*1000:7.3f} ms", flush=True)
    print(f"  pos4.fill (graph 外静态输入)   : {acc['fill']/NSTEP*1000:7.3f} ms", flush=True)
    print(f"  replay CPU 启动               : {acc['replay_cpu']/NSTEP*1000:7.3f} ms", flush=True)
    print(f"  .item() 同步 (含 GPU 等待)    : {acc['item']/NSTEP*1000:7.3f} ms", flush=True)
    print(f"  GPU 侧 graph replay (event)   : {gpu_replay_ms/NSTEP:7.3f} ms", flush=True)
    print(f"  CPU 侧合计 (gather+h2d+fill+replay+item): "
          f"{(acc['gather']+acc['h2d']+acc['fill']+acc['replay_cpu']+acc['item'])/NSTEP*1000:.3f} ms",
          flush=True)
    non_gemm = step_ms - gpu_replay_ms / NSTEP
    print(f"  非 graph GPU 时长 (step - replay_gpu): {non_gemm:.3f} ms", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
