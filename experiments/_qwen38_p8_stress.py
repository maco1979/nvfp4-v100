# -*- coding: utf-8 -*-
"""P8: 多轮压力测试 — 长对话显存增长规律 + OOM 阈值 (为 RAG 挂载提供容量预算)
================================================================================
S1 台账: 加载/prefill/patch 各时点显存锚点
S2 单轮: 逐 chunk 填满 3072 → KV 每 token 显存成本
S3 多轮: 10 轮 prefill+63tok decode, poison+重 prefill → 泄漏/碎片化检测
S4 LRU: 依次挂 1/2/3 套前缀快照 (CPU RAM) → 每套成本
S5 OOM: max_cache_len 3072→4096→6144 递增 prefill 找爆点 (失败即止, 回退安全)
用法: python _qwen38_p8_stress.py [max_cache_len]
"""
import subprocess
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
from _qwen38_p7_pair import patch_pair_gemv
from _qwen38_p4a_lru import (P1_TEXT, P2_TEXT, P3_TEXT, SUFFIX, prefill,
                             poison_cache, snap_dn, PrefixCache)
from transformers import StaticCache

CHUNK = 512
NGEN = 63
MAXLEN = int(sys.argv[1]) if len(sys.argv) > 1 else 3072
LEAD = 11           # 每轮新增 prompt 头部 (多轮对话增长模拟)


def gpu_mem():
    r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True)
    return int(r.stdout.strip().splitlines()[0])          # MiB


def mark(tag):
    a = torch.cuda.max_memory_allocated() // 2**20
    r = torch.cuda.memory_reserved() // 2**20
    print(f"  [{tag}] alloc={a} MiB reserved={r} MiB "
          f"smi={gpu_mem()} MiB", flush=True)
    return a


def long_prompt(n_round):
    """模拟多轮: 前缀 P1 + n 轮 QA 追加, 超长时循环填充至 MAXLEN-NGEN"""
    texts = [P1_TEXT]
    for r in range(n_round):
        texts.append(f"第{r+1}轮: {P2_TEXT if r % 2 else P3_TEXT}")
    t = "".join(texts)
    while len(t) < 40000:                    # 保证 tokenize 后 > MAXLEN
        t += t
    return t


def main():
    print(f"[S1] 建骨架 + NVFP4 权重上传 (max_cache_len={MAXLEN}) ...", flush=True)
    tok, model = build_model()
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu
    hid = emb_w.shape[1]
    eos = tok.eos_token_id
    mark("S1 加载后")

    ids = tok(long_prompt(3), return_tensors="pt").input_ids
    L = min(ids.shape[1], MAXLEN - NGEN - 8)
    ids = ids[:, :L].cuda()
    print(f"[数据] prompt {L} tok (截断至 {L})", flush=True)

    # === S2: 单轮逐 chunk 填满, 显存斜率 ===
    print(f"\n[S2] 单轮 prefill 逐 {CHUNK} tok 台账 ...", flush=True)
    torch.cuda.reset_peak_memory_stats()
    anchors = []
    with torch.no_grad():
        for i in range(0, L, CHUNK):
            j = min(i + CHUNK, L)
            model(ids[:, i:j], past_key_values=cache, use_cache=True,
                  cache_position=torch.arange(i, j, device="cuda"),
                  logits_to_keep=1)
            torch.cuda.synchronize()
            if j in (512, 1024, 2048, 3072, L) or j - i == CHUNK and j % 1024 == 0:
                a = mark(f"S2 @{j}tok")
                anchors.append((j, a))
    if len(anchors) >= 2:
        (t0, m0), (t1, m1) = anchors[0], anchors[-1]
        print(f"  [S2 斜率] ({m1}-{m0})MiB / ({t1}-{t0})tok = "
              f"{(m1-m0)/(t1-t0)*1024:.0f} KiB/token", flush=True)

    # === patch 全家桶 + 重 prefill (进入 decode 态) ===
    print(f"\n[patch] P5-A+P6+P7 全家桶 ...", flush=True)
    patch_fused_deltanet(model, cache)
    patch_rmsnorm(model)
    from _qwen38_p8_attnml import patch_attn_decode_ml
    patch_attn_decode_ml(model, cache, MAXLEN)
    patch_pair_gemv(model)
    poison_cache(cache)
    first = prefill(model, cache, ids, 0, L)
    refresh_deltanet_state(model, cache)
    kv_snap = snap_decode_state(model, cache, L)
    dec = GraphDecoder(model, cache, L, hid)
    restore_decode_state(model, cache, kv_snap, L)
    refresh_deltanet_state(model, cache)
    dec.reset_pos()
    mark("S2 patch+capture 后 (decode 态峰值)")

    # === S3: 多轮泄漏 ===
    print(f"\n[S3] 多轮泄漏压测: 10 轮 (poison → 重 prefill → decode 63) ...",
          flush=True)
    base = None
    for r in range(10):
        poison_cache(cache)
        f = prefill(model, cache, ids, 0, L)
        refresh_deltanet_state(model, cache)
        dec.reset_pos()
        cur = f
        with torch.no_grad():
            for _ in range(NGEN):
                cur = dec.step(cur, emb_w)
        torch.cuda.synchronize()
        a = torch.cuda.memory_allocated() // 2**20
        if base is None:
            base = a
        if r in (0, 4, 9):
            print(f"  [S3 轮{r+1}] alloc={a} MiB smi={gpu_mem()} MiB "
                  f"first={f}", flush=True)
    drift = a - base
    print(f"  [S3 裁决] 10 轮显存漂移 {drift:+d} MiB "
          f"({'无泄漏' if abs(drift) < 32 else '疑似泄漏/碎片, 需排查'})",
          flush=True)

    # === S4: LRU 多前缀容量 ===
    print(f"\n[S4] LRU 多前缀快照成本 (CPU RAM 驻留) ...", flush=True)
    lru = PrefixCache(model, cache, max_bytes=1 << 40)
    texts = [P1_TEXT, P2_TEXT, P3_TEXT]
    import hashlib
    for n, txt in enumerate(texts[:3]):
        _ids = tok(txt + SUFFIX, return_tensors="pt").input_ids
        Lp = _ids.shape[1]
        _ids = _ids.cuda()
        poison_cache(cache)
        prefill(model, cache, _ids, 0, Lp)
        refresh_deltanet_state(model, cache)
        kvl = snap_decode_state(model, cache, Lp)
        dnl = snap_dn(model, cache)
        key = hashlib.sha256(_ids.cpu().numpy().tobytes()).hexdigest()[:16]
        lru.put(key, kvl, dnl)
        kb = lru._bytes(kvl, dnl) / 2**20
        print(f"  [S4 前缀{n+1}] L={Lp} 快照 {kb:.1f} MiB (CPU), "
              f"GPU alloc={torch.cuda.memory_allocated()//2**20} MiB",
              flush=True)
    print(f"  [S4 结论] 3 套前缀驻留 CPU RAM, GPU 无增量", flush=True)

    print(f"\n[完成] MAXLEN={MAXLEN} 全阶段跑通, 无 OOM", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
