# -*- coding: utf-8 -*-
"""P8-S5: 非 3072 max_cache_len 的 attn decode patch (本地复刻版)
================================================================================
_qwen38_infer.patch_attn_decode 硬绑 nvfp4_attn_decode.dll (MAXLEN=3072 编译)。
S5 压测需 6144 — 加载 nvfp4_attn_decode_6144.dll (-DMAXLEN=6144 独立编译,
s_sc 共享数组与 KV stride 均按 6144)。逻辑与原版一致, 仅 DLL 名不同。
"""
import ctypes
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(encoding="utf-8")
from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb


def patch_attn_decode_ml(model, cache, max_len):
    """与 _qwen38_infer.patch_attn_decode 等价, 但按 max_len 选 DLL"""
    dll = ("nvfp4_attn_decode.dll" if max_len == 3072 else
           f"nvfp4_attn_decode_{max_len}.dll")
    lib = ctypes.CDLL(os.path.join(_ROOT, "kernels", dll))
    lib.launch_attn_decode.argtypes = [ctypes.c_void_p] * 6 + \
        [ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    lib.launch_attn_decode.restype = ctypes.c_int
    p = ctypes.c_void_p
    mods = []
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "full_attention":
            continue
        mod, lay = lyr.self_attn, cache.layers[i]
        mod._vf_attn_out = torch.zeros(mod.config.num_attention_heads * mod.head_dim,
                                       dtype=torch.float16, device="cuda")
        if not getattr(mod, "_vf_attn", False):
            mod._vf_attn_orig = mod.forward
        mod._vf_attn = True
        mods.append(mod)

    def fast_fwd(mod, lay):
        def fwd(hidden_states, position_embeddings=None, attention_mask=None,
                past_key_values=None, **kw):
            Tn = hidden_states.shape[1]
            if Tn != 1 or past_key_values is None:
                return mod._vf_attn_orig(
                    hidden_states, position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values, **kw)
            D = mod.head_dim
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, D)
            qg = mod.q_proj(hidden_states).view(*input_shape, -1, D * 2)
            query_states, gate = torch.chunk(qg, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)
            query_states = mod.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = mod.k_norm(mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)
            keys, values = past_key_values.update(key_states, value_states,
                                                   mod.layer_idx)
            st = p(torch.cuda.current_stream().cuda_stream)
            rc = lib.launch_attn_decode(
                p(query_states.data_ptr()), p(keys.data_ptr()),
                p(values.data_ptr()), p(lay.cumulative_length.data_ptr()),
                p(gate.data_ptr()), p(mod._vf_attn_out.data_ptr()),
                max_len, float(mod.scaling), st)
            assert rc == 0, f"attn_decode rc={rc} layer={mod.layer_idx}"
            attn_output = mod._vf_attn_out.view(1, 1, -1)
            return mod.o_proj(attn_output), None
        return fwd

    for mod, lay in zip(mods, [cache.layers[i] for i, l in
                               enumerate(model.model.layers)
                               if l.block_type == "full_attention"]):
        mod.forward = fast_fwd(mod, lay)
    print(f"  [P8-attn] patch 完成 (DLL={dll}, max_len={max_len})", flush=True)
    return mods
