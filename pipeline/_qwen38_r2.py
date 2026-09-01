# -*- coding: utf-8 -*-
"""路线 R2 实测: fused DeltaNet decode kernel vs 原始 torch 回退
流程:
  1) 构模 + prefill + baseline greedy 63 tok (原路径, 重测公平基线)
  2) 重新 prefill, 从 DynamicCache 抓 conv/recurrent 状态 -> 自管静态 buffer
  3) A/B: 同一起始状态, 原路径一步 vs kernel 一步, 比 logits + layer0 状态
  4) 恢复快照, 全部 48 层 patch fast 路径, greedy 63 tok 测速 + token 序列对比
"""
import ctypes
import json
import sys
import time
import types

import numpy as np
import torch

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(encoding="utf-8")
import _qwen38_infer as qi

PROMPT = "用一句话介绍你自己"
NDEC = 63


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
    from transformers import DynamicCache
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    return tok(text, return_tensors="pt").input_ids.cuda()


def greedy(model, first_tok, n, patch_on=None):
    cur = torch.tensor([[first_tok]], device="cuda")
    toks = [first_tok]
    if patch_on is not None:
        set_fast(model, patch_on)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n - 1):
            logits = model(cur, past_key_values=None if False else cur_cache(model)).logits \
                if False else model(cur, past_key_values=CUR_CACHE[0]).logits
            nxt = int(logits[:, -1, :].argmax(dim=-1).item())
            toks.append(nxt)
            cur = torch.tensor([[nxt]], device="cuda")
    torch.cuda.synchronize()
    return toks, time.perf_counter() - t0


CUR_CACHE = [None]


def cur_cache(model):
    return CUR_CACHE[0]


def set_fast(model, on):
    for lyr in model.model.layers:
        if lyr.block_type == "linear_attention":
            lyr.linear_attn._vf_fast = on


def main():
    tok, model = build()
    from transformers import DynamicCache

    # ---- 1) prefill + baseline greedy ----
    ids = make_prompt_ids(tok, PROMPT)
    cache = DynamicCache(config=model.config)
    CUR_CACHE[0] = cache
    with torch.no_grad():
        out = model(ids, past_key_values=cache, use_cache=True)
    first_tok = int(out.logits[:, -1, :].argmax(dim=-1).item())
    set_fast(model, False)
    toks_base, dt_base = greedy(model, first_tok, NDEC)
    print(f"[baseline] {NDEC-1} tok / {dt_base:.2f}s = {(NDEC-1)/dt_base:.2f} tok/s", flush=True)

    # ---- 2) 重新 prefill, 抓线性层状态 ----
    cache2 = DynamicCache(config=model.config)
    CUR_CACHE[0] = cache2
    with torch.no_grad():
        out2 = model(ids, past_key_values=cache2, use_cache=True)
    first2 = int(out2.logits[:, -1, :].argmax(dim=-1).item())
    assert first2 == first_tok

    dn = ctypes.CDLL(os.path.join(_ROOT, "kernels", "nvfp4_dn_fused.dll"))
    dn.launch_dn_fused.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_float, ctypes.c_void_p]
    dn.launch_dn_fused.restype = ctypes.c_int

    lin_layers = []
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "linear_attention":
            continue
        mod = lyr.linear_attn
        lay = cache2.layers[i]
        conv = lay.conv_states[0].reshape(-1)
        rec = lay.recurrent_states[0]                     # (1,48,128,128)
        mod._vf_conv = torch.zeros(10240 * 4, dtype=torch.float16, device="cuda")
        mod._vf_conv.copy_(conv.to(torch.float16))   # (1,10240,4) 布局 c*4+t
        mod._vf_S = rec.transpose(-1, -2).contiguous().float().reshape(-1).clone()
        mod._vf_scr = torch.zeros(10240, dtype=torch.float32, device="cuda")
        mod._vf_out = torch.zeros(6144, dtype=torch.float16, device="cuda")
        mod._vf_eps = float(getattr(mod.norm, "variance_epsilon", 1e-6))
        mod._vf_orig = mod.forward
        mod._vf_fast = False
        lin_layers.append((i, mod))
    print(f"[states] {len(lin_layers)} 线性层状态已抓取 "
          f"(conv {tuple(lin_layers[0][1]._vf_conv.shape)}, "
          f"S {tuple(lin_layers[0][1]._vf_S.shape)})", flush=True)

    # 快照 (A/B 后恢复)
    snap = [(m._vf_conv.clone(), m._vf_S.clone()) for _, m in lin_layers]

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

    for _, mod in lin_layers:
        mod.forward = fast_fwd(mod)

    # ---- 3) A/B: 同一起始状态一步 ----
    cur = torch.tensor([[first_tok]], device="cuda")
    with torch.no_grad():
        # A: 原路径 (更新 cache2)
        la = model(cur, past_key_values=cache2).logits[:, -1, :].float()
        ref_S0 = cache2.layers[lin_layers[0][0]].recurrent_states[0].clone()
    # B: kernel —— 重新 prefill 干净 cache3, 避免 A 步污染 gated-attn KV
    cache3 = DynamicCache(config=model.config)
    CUR_CACHE[0] = cache3
    with torch.no_grad():
        model(ids, past_key_values=cache3, use_cache=True)
        for k, (_, m) in enumerate(lin_layers):
            m._vf_conv.copy_(snap[k][0]); m._vf_S.copy_(snap[k][1])
        set_fast(model, True)
        lb = model(cur, past_key_values=cache3).logits[:, -1, :].float()
        set_fast(model, False)
        torch.cuda.synchronize()
    d = (la - lb).abs()
    print(f"[A/B logits] max_abs={d.max().item():.4f} mean_abs={d.mean().item():.5f} | "
          f"argmax A={int(la.argmax())} B={int(lb.argmax())}", flush=True)
    got_S = lin_layers[0][1]._vf_S.view(48, 128, 128)
    ref_S = ref_S0[0].transpose(-1, -2)
    print(f"[A/B state0] max_abs={(got_S - ref_S).abs().max().item():.5f}", flush=True)

    # ---- 4) fast greedy (重新 prefill 干净 cache4) ----
    cache4 = DynamicCache(config=model.config)
    CUR_CACHE[0] = cache4
    with torch.no_grad():
        model(ids, past_key_values=cache4, use_cache=True)
    for k, (_, m) in enumerate(lin_layers):
        m._vf_conv.copy_(snap[k][0]); m._vf_S.copy_(snap[k][1])
    set_fast(model, True)
    toks_fast, dt_fast = greedy(model, first_tok, NDEC)
    match = sum(a == b for a, b in zip(toks_base, toks_fast))
    print(f"[fast R2] {NDEC-1} tok / {dt_fast:.2f}s = {(NDEC-1)/dt_fast:.2f} tok/s | "
          f"token 匹配 {match}/{NDEC}", flush=True)
    print("[fast 输出]", tok.decode(toks_fast)[:180], flush=True)


if __name__ == "__main__":
    main()
