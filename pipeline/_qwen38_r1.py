# -*- coding: utf-8 -*-
"""路线 R1 实测: StaticCache + CUDA Graph 捕获 decode 单步
==========================================================
测试矩阵 (同一次 build, 公平对比):
  0) baseline  : DynamicCache 原路径 (R2 已测 5.77, 复测核对)
  1) eager     : StaticCache 无 graph —— 分离 StaticCache 自身影响
  2) graph     : StaticCache + CUDA Graph (forward+argmax+pos+=1 全捕获)
  3) graph+fused: graph 包住 R2 fused DeltaNet kernel (R1+R2 组合)

关键设计:
  - QEmbedding CPU gather 在 graph 外, pinned buffer H2D 进静态输入
  - StaticLayer.cumulative_length 是 GPU tensor + add_ 原地 —— graph 内自动递增
  - KV index_copy_ / conv / recurrent 全部原地更新, 地址静态
  - warmup(side stream) 会污染状态 → capture 后恢复快照再正式测速
"""
import ctypes
import json
import sys
import time

import numpy as np
import torch

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import _qwen38_infer as qi

PROMPT = "用一句话介绍你自己"
NDEC = 63
MAXLEN = 160          # prefill(~20) + 63 + 余量


def build():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights

    tok = AutoTokenizer.from_pretrained(qi.MODEL_DIR)
    cfg = AutoConfig.from_pretrained(qi.MODEL_DIR)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg.text_config, dtype=torch.float16)
    index = json.load(open(qi.IDX, encoding="utf-8"))
    misc = np.load(qi.MISC)
    embed = np.load(qi.EMB, mmap_mode="r")
    model.model.embed_tokens = qi.QEmbedding(torch.from_numpy(np.asarray(embed)))
    qi.patch_linears(model, index, misc)
    model.to_empty(device="cuda")
    qi.fill_rest(model, misc, embed)
    qi.rebuild_rope(model)
    model.eval()
    return tok, model


def make_prompt_ids(tok, prompt):
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, return_tensors="pt").input_ids.cuda()


def lin_attn_indices(model):
    return [i for i, l in enumerate(model.model.layers)
            if l.block_type == "linear_attention"]


def snap_static(model, cache, L):
    """KV 前缀 + linear 层 conv/recurrent + 各层 cumulative_length"""
    kv, lin = [], []
    for i, lyr in enumerate(model.model.layers):
        lay = cache.layers[i]
        if lyr.block_type == "linear_attention":
            lin.append((lay.conv_states[0].clone(),
                        lay.recurrent_states[0].clone()))
        else:
            kv.append((lay.keys[:, :, :L].clone(), lay.values[:, :, :L].clone(),
                       lay.cumulative_length.clone()))
    return kv, lin


def restore_static(model, cache, snap, L):
    kv, lin = snap
    ki = li = 0
    for i, lyr in enumerate(model.model.layers):
        lay = cache.layers[i]
        if lyr.block_type == "linear_attention":
            lay.conv_states[0].copy_(lin[li][0])
            lay.recurrent_states[0].copy_(lin[li][1])
            li += 1
        else:
            k, v, c = kv[ki]
            lay.keys[:, :, :L].copy_(k)
            lay.values[:, :, :L].copy_(v)
            lay.cumulative_length.copy_(c)
            ki += 1


def embed_cpu_gather(model):
    """(V,6144) fp16 CPU embedding 矩阵"""
    return model.model.embed_tokens.weight_cpu


def greedy_dyn(model, cache, first, n):
    """baseline: DynamicCache 原路径"""
    emb_w = embed_cpu_gather(model)
    toks = [first]
    cur = torch.tensor([[first]], device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n - 1):
            logits = model(cur, past_key_values=cache, use_cache=True).logits
            nxt = int(logits[:, -1, :].argmax(dim=-1).item())
            toks.append(nxt)
            cur = torch.tensor([[nxt]], device="cuda")
    torch.cuda.synchronize()
    return toks, time.perf_counter() - t0


def greedy_static_eager(model, cache, first, n, pos0, emb_w):
    """StaticCache 无 graph —— 每步 inputs_embeds + cache_position"""
    toks = [first]
    pos = torch.tensor([pos0], device="cuda")
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n - 1):
            emb = emb_w[toks[-1]].view(1, 1, -1).to("cuda", non_blocking=False)
            logits = model(inputs_embeds=emb, past_key_values=cache,
                           cache_position=pos, use_cache=True).logits
            nxt = int(logits[:, -1, :].argmax(dim=-1).item())
            toks.append(nxt)
            pos += 1
    torch.cuda.synchronize()
    return toks, time.perf_counter() - t0


def run_graph(model, cache, first, n, pos0, emb_w, snap, L, tag, extra_restore=None):
    """warmup + capture + 恢复快照 + 测速 (extra_restore: fused 层自管状态恢复)"""
    HID = emb_w.shape[1]                            # hidden_size (5120)
    static_embeds = torch.zeros(1, 1, HID, dtype=torch.float16, device="cuda")
    static_pos = torch.tensor([pos0], device="cuda")
    static_next = torch.zeros(1, dtype=torch.long, device="cuda")
    pin = torch.empty(HID, dtype=torch.float16, pin_memory=True)

    def one_step():
        out = model(inputs_embeds=static_embeds, past_key_values=cache,
                    cache_position=static_pos, use_cache=True)
        static_next.copy_(out.logits[:, -1, :].argmax(dim=-1))
        static_pos.add_(1)      # 原地: += 会重绑定闭包变量

    # warmup: side stream 3 步 (污染状态, 之后恢复)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s), torch.no_grad():
        for _ in range(3):
            one_step()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g), torch.no_grad():
        one_step()

    # 恢复快照
    restore_static(model, cache, snap, L)
    if extra_restore is not None:
        extra_restore()
    static_pos.fill_(pos0)
    torch.cuda.synchronize()

    def replay(tok_id):
        pin.copy_(emb_w[tok_id])
        static_embeds.copy_(pin.view(1, 1, -1), non_blocking=True)
        g.replay()
        return int(static_next.item())

    toks = [first]
    nxt = replay(first)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n - 1):
        toks.append(nxt)
        nxt = replay(nxt)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"[{tag}] {NDEC-1} tok / {dt:.2f}s = {(NDEC-1)/dt:.2f} tok/s", flush=True)
    return toks, dt


def patch_fused(model, cache, dn_path=os.path.join(_ROOT, "kernels", "nvfp4_dn_fused.dll")):
    """R2 fused kernel: linear 层换 fast forward, 状态转自管 buffer"""
    dn = ctypes.CDLL(dn_path)
    dn.launch_dn_fused.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_float, ctypes.c_void_p]
    dn.launch_dn_fused.restype = ctypes.c_int
    mods = []
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "linear_attention":
            continue
        mod = lyr.linear_attn
        lay = cache.layers[i]
        conv = lay.conv_states[0].reshape(-1)
        rec = lay.recurrent_states[0]
        mod._vf_conv = torch.zeros(10240 * 4, dtype=torch.float16, device="cuda")
        mod._vf_conv.copy_(conv.to(torch.float16))
        mod._vf_S = rec.transpose(-1, -2).contiguous().float().reshape(-1).clone()
        mod._vf_scr = torch.zeros(10240, dtype=torch.float32, device="cuda")
        mod._vf_out = torch.zeros(6144, dtype=torch.float16, device="cuda")
        mod._vf_eps = float(getattr(mod.norm, "variance_epsilon", 1e-6))
        mod._vf_orig = mod.forward
        mod._vf_fast = True
        mods.append(mod)

    def fast_fwd(mod):
        def fwd(hidden_states, cache_params=None, attention_mask=None, **kw):
            if hidden_states.shape[1] != 1 or not mod._vf_fast:
                return mod._vf_orig(hidden_states, cache_params=cache_params,
                                    attention_mask=attention_mask, **kw)
            mixed = mod.in_proj_qkv(hidden_states)
            z = mod.in_proj_z(hidden_states)
            b = mod.in_proj_b(hidden_states)
            a = mod.in_proj_a(hidden_states)
            st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            p = ctypes.c_void_p
            rc = dn.launch_dn_fused(
                p(mixed.data_ptr()), p(z.data_ptr()), p(b.data_ptr()), p(a.data_ptr()),
                p(mod.conv1d.weight.data_ptr()), p(mod._vf_conv.data_ptr()),
                p(mod.A_log.data_ptr()), p(mod.dt_bias.data_ptr()),
                p(mod.norm.weight.data_ptr()), p(mod._vf_S.data_ptr()),
                p(mod._vf_scr.data_ptr()), p(mod._vf_out.data_ptr()),
                mod._vf_eps, st)
            assert rc == 0, f"dn_fused rc={rc}"
            return mod.out_proj(mod._vf_out.view(1, 1, -1))
        return fwd

    for mod in mods:
        mod.forward = fast_fwd(mod)
    return mods


def main():
    from transformers import DynamicCache, StaticCache

    tok, model = build()
    ids = make_prompt_ids(tok, PROMPT)
    L = ids.shape[1]
    emb_w = embed_cpu_gather(model)

    # ---- 0) baseline DynamicCache ----
    cache0 = DynamicCache(config=model.config)
    with torch.no_grad():
        out0 = model(ids, past_key_values=cache0, use_cache=True)
    first = int(out0.logits[:, -1, :].argmax(dim=-1).item())
    toks_base, dt0 = greedy_dyn(model, cache0, first, NDEC)
    print(f"[baseline dyn] {NDEC-1} tok / {dt0:.2f}s = {(NDEC-1)/dt0:.2f} tok/s",
          flush=True)

    # ---- 1) StaticCache eager ----
    sc = StaticCache(config=model.config, max_cache_len=MAXLEN)
    with torch.no_grad():
        out1 = model(ids, past_key_values=sc, use_cache=True)
    first1 = int(out1.logits[:, -1, :].argmax(dim=-1).item())
    print(f"[static prefill] first_tok={first1} (dyn={first}) 一致={first1 == first}",
          flush=True)
    snap = snap_static(model, sc, L)   # prefill 后立即抓 (eager greedy 会推进状态)
    toks_e, dt_e = greedy_static_eager(model, sc, first1, NDEC, L, emb_w)
    print(f"[eager static] {NDEC-1} tok / {dt_e:.2f}s = {(NDEC-1)/dt_e:.2f} tok/s",
          flush=True)
    m = sum(a == b for a, b in zip(toks_base, toks_e))
    print(f"[eager static] token 匹配 {m}/{NDEC}", flush=True)

    # ---- 2) StaticCache + CUDA Graph ----
    toks_g, dt_g = run_graph(model, sc, first1, NDEC, L, emb_w, snap, L, "graph R1")
    m = sum(a == b for a, b in zip(toks_base, toks_g))
    print(f"[graph R1] token 匹配 {m}/{NDEC}", flush=True)
    print("[graph R1 输出]", tok.decode(toks_g)[:120], flush=True)

    # ---- 3) graph + R2 fused (组合) ----
    # 重置 StaticCache (重新 prefill 抓干净状态), 再 patch fused
    sc2 = StaticCache(config=model.config, max_cache_len=MAXLEN)
    with torch.no_grad():
        model(ids, past_key_values=sc2, use_cache=True)
    mods = patch_fused(model, sc2)
    snap2 = snap_static(model, sc2, L)
    # fused 层状态在 _vf_* 自管 buffer, 快照扩展
    vf_snap = [(md._vf_conv.clone(), md._vf_S.clone()) for md in mods]

    def restore_vf():
        for md, (c, s) in zip(mods, vf_snap):
            md._vf_conv.copy_(c)
            md._vf_S.copy_(s)

    toks_gf, dt_gf = run_graph(model, sc2, first1, NDEC, L, emb_w, snap2, L,
                               "graph R1+R2", extra_restore=restore_vf)
    print(f"[graph R1+R2] token 匹配 "
          f"{sum(a == b for a, b in zip(toks_base, toks_gf))}/{NDEC}", flush=True)
    print("[graph R1+R2 输出]", tok.decode(toks_gf)[:120], flush=True)


if __name__ == "__main__":
    main()
