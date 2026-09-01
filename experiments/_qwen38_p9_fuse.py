# -*- coding: utf-8 -*-
"""P9: RoPE QK + MLP silu·mul 融合 (nvfp4_rope_silu.dll, sm_70)
================================================================================
背景 (P6 profile decode 每步 elementwise 残余 ~2.0ms):
  1. 16 层 full_attention 的 apply_rotary_pos_emb: torch 链 ~10 kernel/层
     (chunk/cat/mul/add/广播), ~0.8ms/步 → 1 launch/层
  2. 全部层 MLP 的 silu(gate)*up: 2 kernel/层, ~0.5ms/步 → 1 launch/层
位精确链 (逐 op 对齐 torch CUDA opmath=fp32):
  RoPE: t1=f16(q·cos) t2=f16(rot(q)·sin) out=f16(t1+t2)
  silu: s=f16(g/(1+expf(-g))) (除法, 非 1/(...)*g 乘倒数, ulp 差异)
        out=f16(f32(s)·f32(u))
  expf 软件精确版, 禁用 __expf (发现64教训)
注入方式:
  - MLP: 替换 Qwen3_5MLP.forward 的 T==1 分支
  - RoPE: monkey-patch 模块级 apply_rotary_pos_emb; fast_fwd 闭包是
    函数级 from-import → patch 后必须重新调用 patch_attn_decode() 重建
    闭包才生效 (调用顺序由验收脚本 _qwen38_p9_acc.py 保证)
T>1 (prefill/MTP) 回退原路径。
用法: 由验收脚本调用
"""
import ctypes
import os
import sys

import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.stdout.reconfigure(encoding="utf-8")

D = 256            # head_dim (Qwen3.8 full_attention, 全旋转)
_lib = None


def _get_lib():
    global _lib
    if _lib is None:
        _lib = ctypes.CDLL(os.path.join(_ROOT, "kernels", "nvfp4_rope_silu.dll"))
        _lib.launch_rope_qk.argtypes = [ctypes.c_void_p] * 6 + \
            [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        _lib.launch_rope_qk.restype = ctypes.c_int
        _lib.launch_silu_mul.argtypes = [ctypes.c_void_p] * 3 + \
            [ctypes.c_int, ctypes.c_void_p]
        _lib.launch_silu_mul.restype = ctypes.c_int
    return _lib


def patch_rope(model):
    """monkey-patch qwen3_5.apply_rotary_pos_emb → T=1 走 rope_qk_kernel"""
    import transformers.models.qwen3_5.modeling_qwen3_5 as m
    if getattr(m.apply_rotary_pos_emb, "_vf_p9", False):
        return False
    orig = m.apply_rotary_pos_emb
    lib = _get_lib()
    P = ctypes.c_void_p

    def fused_rope(q, k, cos, sin, unsqueeze_dim=1):
        # T=1 + 全旋转 + 每头连续 (q_norm(...).transpose(1,2) 布局) 才融合
        if (unsqueeze_dim == 1 and q.dim() == 4 and q.shape[2] == 1
                and q.shape[3] == D and k.shape[3] == D
                and q.dtype == torch.float16 and cos.dtype == torch.float16
                and cos.shape[-1] == D and cos.stride(-1) == 1
                and q.stride(3) == 1 and q.stride(1) == D
                and k.stride(3) == 1 and k.stride(1) == D):
            hq, hkv = q.shape[1], k.shape[1]
            q_out = torch.empty(q.shape, dtype=torch.float16, device=q.device)
            k_out = torch.empty(k.shape, dtype=torch.float16, device=k.device)
            st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            rc = lib.launch_rope_qk(
                P(q.data_ptr()), P(k.data_ptr()),
                P(cos.data_ptr()), P(sin.data_ptr()),
                P(q_out.data_ptr()), P(k_out.data_ptr()),
                hq, hkv, st)
            assert rc == 0, f"rope_qk rc={rc}"
            return q_out, k_out
        return orig(q, k, cos, sin, unsqueeze_dim)

    fused_rope._vf_p9 = True
    m.apply_rotary_pos_emb = fused_rope
    print("  [P9] RoPE QK 融合: apply_rotary_pos_emb → 1 launch/层 "
          "(原 torch 链 ~10 kernel/层)", flush=True)
    return True


def patch_mlp_silu_mul(model):
    """全部 Qwen3_5MLP.forward 的 T==1 分支 → silu_mul_kernel"""
    lib = _get_lib()
    P = ctypes.c_void_p
    n = 0
    for lyr in model.model.layers:
        mlp = lyr.mlp
        if getattr(mlp, "_vf_p9", False):
            continue
        orig = mlp.forward

        def fwd(x, _mlp=mlp, _orig=orig, _lib=lib, _P=P):
            if x.shape[1] == 1:
                g = _mlp.gate_proj(x)                  # (1,1,I) fp16
                u = _mlp.up_proj(x)                    # (1,1,I) fp16
                out = torch.empty(g.shape, dtype=torch.float16,
                                  device=g.device)
                st = ctypes.c_void_p(
                    torch.cuda.current_stream().cuda_stream)
                rc = _lib.launch_silu_mul(
                    _P(g.data_ptr()), _P(u.data_ptr()), _P(out.data_ptr()),
                    g.numel(), st)
                assert rc == 0, f"silu_mul rc={rc}"
                return _mlp.down_proj(out)
            return _orig(x)

        mlp.forward = fwd
        mlp._vf_p9 = True
        n += 1
    print(f"  [P9] MLP silu·mul 融合: {n} 层 × 1 launch "
          f"(原 silu+mul 2 kernel/层)", flush=True)
    return n


def patch_rope_silu(model):
    a = patch_rope(model)
    b = patch_mlp_silu_mul(model)
    return a, b
