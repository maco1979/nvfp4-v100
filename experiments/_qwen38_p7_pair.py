# -*- coding: utf-8 -*-
"""P7-2: 成对 GEMV 融合 — 小 shape GEMV 欠占用 SM 的治理
================================================================================
背景 (P7-1 隔离 bench 裁决): in-graph 663GB/s vs 隔离加权 722GB/s, 差距 8% 是
干扰, 但更主要是尺寸效应——48×5120 微型 GEMV (48 warp, 80 SM 上占 0.6 warp/SM)
仅 18GB/s, 1024×5120 也只 300GB/s。合计 ~0.75ms/步纯欠占用浪费。
方法: 同输入成对调用 (每层 in_proj_b→in_proj_a, k_proj→v_proj) 拼连续 buffer,
一次 launch 算 2M 行 (96 / 2048)。行独立计算 → 与两次 M 行 launch 逐位相同。
  - 首成员 (lead): T=1 时无条件算整组, 返回自己的行切片
  - 次成员: 校验 data_ptr 一致 → 取切片; 异常序 fallback 偏移指针独立算
  - T>1 (prefill): 走 super(), 次成员 d_p/d_s 已是偏移指针, 语义不变
验收: 63 token 逐位 + tok/s (P6 基线 33.9)
用法: 由验收脚本 _qwen38_p7_pair_acc.py 调用
"""
import ctypes
import sys

import torch

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")
import _qwen38_infer as _qi
from _qwen38_infer import GEMV, QLinear


class _PairLinear(QLinear):
    """成对融合成员: forward T=1 时走组共享 launch / 切片"""

    def forward(self, x):
        K = self.K
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K).contiguous()
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
        if x2.shape[0] == 1:
            g = self._group
            if self._lead:                      # 首成员: 无条件算整组 (无跨步缓存)
                outg = g.compute(x2)
                out = outg[:, self._row_off:self._row_off + self.M]
                if self.bias is not None:
                    out = out + self.bias
                return out.reshape(*lead, self.M)
            if g.out is not None and g.xptr == x2.data_ptr():
                out = g.out[:, self._row_off:self._row_off + self.M]
                if self.bias is not None:
                    out = out + self.bias
                return out.reshape(*lead, self.M)
            # 调用序异常 → 独立路径 (self.d_p 已含行偏移, 数值不变)
        return super().forward(x)


class _Group:
    __slots__ = ("d_p", "d_s", "K", "M_total", "out", "xptr")

    def __init__(self, d_p, d_s, K, M_total):
        self.d_p, self.d_s, self.K, self.M_total = d_p, d_s, K, M_total
        self.out, self.xptr = None, 0

    def compute(self, x2):
        out = torch.empty((1, self.M_total), dtype=torch.float16,
                          device=x2.device)
        stream = torch.cuda.current_stream().cuda_stream
        rc = _qi.GEMV_T1(self.d_p, self.d_s, x2.data_ptr(), out.data_ptr(),
                         self.M_total, self.K, ctypes.c_void_p(stream))
        assert rc == 0, f"gemv(group) rc={rc}"
        self.out, self.xptr = out, x2.data_ptr()
        return out


# 模块属性名 → (首调用 lead, 后调用 follower)。顺序来自模型 forward 实测:
#   linear_attn: b = in_proj_b(x); a = in_proj_a(x)   → lead=b
#   self_attn:   k_proj(x) ... v_proj(x)               → lead=k
PAIRS = {"in_proj_b": ("in_proj_a", True), "in_proj_a": ("in_proj_b", False),
         "k_proj": ("v_proj", True), "v_proj": ("k_proj", False)}


def patch_pair_gemv(model):
    """build_model 后调用: 把成对小 GEMV 换成共享 buffer 的 _PairLinear"""
    done = set()
    n_pair = 0
    for pname, mod in list(model.named_modules()):
        if not isinstance(mod, QLinear) or pname in done:
            continue
        parts = pname.split(".")
        attr = parts[-1]
        if attr not in PAIRS:
            continue
        sib_attr, is_lead = PAIRS[attr]
        if not is_lead:                    # 只从 lead 侧处理一对
            continue
        parent = model.get_submodule(".".join(parts[:-1]))
        sib = getattr(parent, sib_attr, None)
        if not isinstance(sib, QLinear) or sib.K != mod.K:
            continue
        if mod.bias is not None or sib.bias is not None:
            continue                        # 有 bias 不融合 (T=1 原路径无 bias 处理)
        M1, M2, K = mod.M, sib.M, mod.K
        pb1, sb1 = M1 * K // 2, M1 * K // 16
        pb2, sb2 = M2 * K // 2, M2 * K // 16
        d_p = GEMV.gpu_malloc(pb1 + pb2)
        d_s = GEMV.gpu_malloc(sb1 + sb2)
        assert d_p and d_s, "gpu_malloc(pair) failed"
        # D2H 回读旧权重 → H2D 拼进连续 buffer (小权重, MB 级往返)
        host = torch.empty(pb1 + pb2, dtype=torch.uint8)
        assert GEMV.gpu_memcpy_d2h(host[:pb1].numpy().ctypes.data, mod.d_p, pb1) == 0
        assert GEMV.gpu_memcpy_d2h(host[pb1:].numpy().ctypes.data, sib.d_p, pb2) == 0
        assert GEMV.gpu_memcpy_h2d(d_p, host.numpy().ctypes.data, pb1 + pb2) == 0
        hs = torch.empty(sb1 + sb2, dtype=torch.uint8)
        assert GEMV.gpu_memcpy_d2h(hs[:sb1].numpy().ctypes.data, mod.d_s, sb1) == 0
        assert GEMV.gpu_memcpy_d2h(hs[sb1:].numpy().ctypes.data, sib.d_s, sb2) == 0
        assert GEMV.gpu_memcpy_h2d(d_s, hs.numpy().ctypes.data, sb1 + sb2) == 0
        GEMV.gpu_free(mod.d_p)
        GEMV.gpu_free(mod.d_s)
        GEMV.gpu_free(sib.d_p)
        GEMV.gpu_free(sib.d_s)

        g = _Group(d_p, d_s, K, M1 + M2)
        q1 = _PairLinear(d_p, d_s, M1, K, bias=None)
        q1._group, q1._row_off, q1._lead = g, 0, True
        q2 = _PairLinear(d_p + pb1, d_s + sb1, M2, K, bias=None)
        q2._group, q2._row_off, q2._lead = g, M1, False
        setattr(parent, attr, q1)           # lead (in_proj_b / k_proj)
        setattr(parent, sib_attr, q2)      # follower (in_proj_a / v_proj)
        done.update({pname, ".".join(parts[:-1] + [sib_attr])})
        n_pair += 1
    print(f"  [P7-2] 融合 {n_pair} 对小 GEMV "
          f"(launch 数 -{n_pair}/步)", flush=True)
    return n_pair
