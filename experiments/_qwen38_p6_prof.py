# -*- coding: utf-8 -*-
"""P6-2b: eager 模式 + torch.profiler 剖析 decode 步内核分布 (v2)
======================================================================
v1 失败: 进程内先 CUDA Graph capture → CUPTI 静默失效 (0 事件)。
v2: 全程无 capture —— prefill → patch → refresh → eager 前向。
    分析用 key_averages() 聚合 (无需切步边界, profiler 只包 NSTEP 步)。
裁决目标:
  1. 每步 v8 GEMV 数 (NCU 解析 497 假设 vs 表中 332 矛盾)
  2. v3.1 后每步内核分布 (与 _ncu_parse_p6.py 同口径归类)
  3. v8 单核时长 (裁决 NCU 两次运行 29μs vs 43.6μs 的绝对值矛盾)
用法: python _qwen38_p6_prof.py
"""
import sys
from collections import defaultdict

import torch
from torch.profiler import ProfilerActivity, profile

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")
from _qwen38_infer import (build_model, ensure_kv_stage,
                           patch_fused_deltanet, patch_attn_decode,
                           patch_rmsnorm, refresh_deltanet_state)
from transformers import StaticCache

CHUNK = 512
NSTEP = 5
MAXLEN = 3072
PROMPT = "用一句话解释什么是块缩点浮点量化。"


def main():
    print("[1] 建骨架 + NVFP4 权重上传 ...", flush=True)
    tok, model = build_model()
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu

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

    patch_fused_deltanet(model, cache)
    patch_rmsnorm(model)                     # P5-A: norm 融合
    patch_attn_decode(model, cache)          # P6: attn v3.1 kernel
    refresh_deltanet_state(model, cache)
    print(f"[2] prefill {L} tok + patch 完成, eager 前向 {NSTEP} 步 ...",
          flush=True)

    pos = torch.tensor([L], device="cuda")           # cache_position
    pos4 = torch.zeros(4, 1, 1, dtype=torch.long, device="cuda")
    static_embeds = torch.zeros(1, 1, emb_w.shape[1],
                                 dtype=torch.float16, device="cuda")

    def eager_step(t):
        static_embeds.copy_(emb_w[t].view(1, 1, -1).to("cuda", torch.float16))
        pos4.fill_(pos.item())
        with torch.no_grad():
            o = model(inputs_embeds=static_embeds, past_key_values=cache,
                      cache_position=pos, use_cache=True, position_ids=pos4)
            nxt = o.logits[:, -1, :].argmax(dim=-1)
            pos.add_(1)
        return int(nxt.item())

    for _ in range(2):                       # warmup
        cur = eager_step(cur)
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for _ in range(NSTEP):
            cur = eager_step(cur)
        torch.cuda.synchronize()

    # 优先: 直接取 device kernel 事件 (含 ctypes/DLL 外部 launch 的自定义 kernel)
    kevs = [e for e in prof.events()
            if e.device_type == torch.autograd.DeviceType.CUDA
            and e.self_device_time_total > 0]
    print(f"[3] kernel 事件 {len(kevs)} 条", flush=True)
    agg = defaultdict(lambda: [0.0, 0])
    for e in kevs:
        agg[e.name][0] += e.self_device_time_total
        agg[e.name][1] += 1
    tot = sum(a[0] for a in agg.values())
    rows = [(n, a[0], a[1]) for n, a in agg.items()]
    if tot == 0:
        # 回退: CPU op 聚合 (self_device_time 含子 kernel)
        ka = prof.key_averages()
        rows = []
        for e in ka:
            t = getattr(e, "self_device_time_total", 0) or getattr(
                e, "self_cuda_time_total", 0) or 0
            if t and t > 0 and e.count > 0 and getattr(e, "device_type", None) is not None:
                rows.append((e.key, t, e.count))
        tot = sum(r[1] for r in rows)
        print("[warn] kernel 事件为 0, 回退 CPU op 聚合 (DLL kernel 可能不可见)")
    # 只保留 GPU 内核行: 过滤掉纯 CPU 行 (无 device time 的)
    tot = sum(r[1] for r in rows)
    print(f"\n[3] GPU self-time 合计 {tot/1e3:.2f} ms / {NSTEP} 步 → "
          f"每步 {tot/NSTEP/1e3:.2f} ms (eager, 不含 launch gap)")
    if tot == 0:
        print("[fail] 仍未取到 GPU 时间")
        return 1
    print(f"{'内核':<80} {'次/步':>6} {'μs/步':>9} {'占比':>6}")
    print("-" * 108)
    for name, t, c in sorted(rows, key=lambda r: -r[1])[:26]:
        print(f"{name[:78]:<80} {c/NSTEP:>6.0f} {t/NSTEP:>9.1f} {100*t/tot:>5.1f}%")

    cat = defaultdict(lambda: [0.0, 0])
    for name, t, c in rows:
        if "gemv_v8" in name:
            cat["GEMV(v8)"][:] = [cat["GEMV(v8)"][0] + t, cat["GEMV(v8)"][1] + c]
        elif "dn_rec" in name or "dn_conv" in name:
            cat["DN融合"][:] = [cat["DN融合"][0] + t, cat["DN融合"][1] + c]
        elif "attn_decode" in name:
            cat["attn(v3.1)"][:] = [cat["attn(v3.1)"][0] + t, cat["attn(v3.1)"][1] + c]
        elif "rmsnorm" in name:
            cat["rmsnorm(v_fused)"][:] = [cat["rmsnorm(v_fused)"][0] + t, cat["rmsnorm(v_fused)"][1] + c]
        elif "elementwise" in name or "FillFunctor" in name or "CatArray" in name:
            cat["torch elementwise"][:] = [cat["torch elementwise"][0] + t, cat["torch elementwise"][1] + c]
        elif "reduce" in name.lower() or "ArgMax" in name:
            cat["torch reduce"][:] = [cat["torch reduce"][0] + t, cat["torch reduce"][1] + c]
        else:
            cat["其他"][:] = [cat["其他"][0] + t, cat["其他"][1] + c]
    print("\n===== 归类 (每步, eager) =====")
    for k, (v, c) in sorted(cat.items(), key=lambda kv: -kv[1][0]):
        print(f"{k:<28} {v/NSTEP/1e3:>8.1f} μs  {100*v/tot:>5.1f}%  "
              f"({c/NSTEP:.0f} 次/步)")

    nv8 = cat["GEMV(v8)"][1] / NSTEP
    tv8 = cat["GEMV(v8)"][0] / NSTEP
    print(f"\n[v8 核验] 每步 v8 GEMV = {nv8:.1f} 次 (NCU 假设 497 / 表 332 → 本值裁决)")
    if nv8:
        print(f"[v8 时长] 平均 {tv8/nv8:.1f} μs/核 (NCU 两次 29.0 vs 43.6 μs 裁决)")
    print(f"[4] 完成, 末 token = {cur}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
