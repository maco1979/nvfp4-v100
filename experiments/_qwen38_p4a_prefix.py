# -*- coding: utf-8 -*-
"""P4a: 前缀 KV+DeltaNet 状态跨请求复用原型 (单机版 RadixAttention 第一步)
=====================================================================
场景: 固定前缀 (system prompt/文档) + 变化后缀的多轮请求。
机制: 前缀 prefill 一次 → 快照 (14 层 attention KV + 48 层 DeltaNet
      conv/recurrent + cumulative_length) 驻 CPU → 后续请求 restore 后
      仅 prefill 后缀, 跳过前缀重复计算 ("推理税")。
约束: 前缀长度必须是 CHUNK(512) 倍数 — 保证 DeltaNet 分块 scan 的块边界
      与全量 prefill 完全一致 (bit-exact 前提)。
验收: Run3(冷+快照) 与 Run1(baseline A) 63 token 逐位一致;
      Run4(热=restore+仅后缀) 与 Run2(baseline B) 63 token 逐位一致;
      打毒测试: run 间 KV 填 12345.0, restore 遗漏必致 token 发散。
用法: python _qwen38_p4a_prefix.py
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
                           refresh_deltanet_state, GraphDecoder)
from transformers import StaticCache

CHUNK = 512
NGEN = 63
MAXLEN = 3072
L_PRE_TARGET = 1024          # 前缀 token 数 (向下取整到 CHUNK 倍数)

PREFIX_TEXT = (
    "VORTEX FLAME 项目 NVFP4 研究技术文档。本研究完整解析了 Blackwell NVFP4 块缩放"
    "四比特浮点格式, 并在 Tesla V100 上手工实现了全链路软件模拟。NVFP4 由 E2M1 浮点"
    "元素格式与 E8M0 块缩放组成: 每个元素四比特, 包含一位符号、两位指数、一位尾数, "
    "非零码点为零点五、一点零、一点五、二点零、三点零、四点零、六点零共七个; 每十六个"
    "连续元素共享一个八比特纯指数块缩放, 数值为二的幂次。最终数值等于元素值乘以块缩放。"
    "研究经历了十个阶段。阶段一完成格式研究与 Python 编解码参考实现, 小矩阵验证全部"
    "通过, 校准矩阵余弦相似度达到零点九九以上。阶段二开发单缓冲融合反量化 GEMM 内核, "
    "在驱动损坏期间用纯 Python 模拟器完成前置验证。阶段三实现双缓冲二阶段内核并完成"
    "V100 实测, 期间发现反量化实测开销仅百分之六点六, 推翻了理论预估的百分之三十到"
    "四十五; 块缩放编码上限修正为一百二十七, 避免指数二百五十五溢出为无穷大。阶段四"
    "完成 DiT 视频模型完整基准测试, 内核经五代优化从朴素版本演进到 TensorCore 版本, "
    "大批次吞吐达到三十三到三十八万亿次浮点运算每秒。阶段五完成与 Blackwell 硬件的"
    "差距分析: 峰值对峰值二百三十六倍, 但格式数值语义完全等价, 编码器与验收方法论"
    "百分之百可迁移。阶段六将 Qwen3.8 二十七B 模型以十四点四二GB 塞进十六GB 显存, "
    "权重压缩比零点二八精确贴合理论值。阶段七完成两条优化路线的对决: CUDA Graph "
    "根治 CPU 调度瓶颈, 融合内核消除数据类型转换, 组合后吞吐提升三点七七倍。阶段八"
    "完成 GEMV 带宽优化与投机采样对决, 查表方案达到七百九十二GB 每秒, 但投机采样因"
    "接受率天花板不足而关闭。阶段九跑通八K 上下文, 采用分块预填充与三层笔记式架构。"
    "阶段十完成视觉塔量化验证, 确定浅层保留高精度、深层整数量化的混合方案。整个研究"
    "沉淀了三十六条知识库条目, 其中错误隔离七条, 全部带原始错误文本与定位方法。"
    "核心工程结论: 累加器必须使用三十二位浮点; 块缩放编码必须钳位到二百五十四以内; "
    "内核正确性验收基准一律使用六十四位浮点真值; 优化必须以 token 一致为红线, 达不到"
    "就回退上一最佳版本。这些经验对未来任何量化推理工程都有直接参考价值。"
)
SUFFIX_A = "请用三句话总结这份文档的核心内容。"
SUFFIX_B = "文档中提到的权重显存压缩比和实测吞吐分别是多少? 请详细说明。"


def prefill(model, cache, ids, start, end):
    """分块 prefill [start, end), 返回末 token 的 argmax"""
    first = None
    with torch.no_grad():
        for i in range(start, end, CHUNK):
            j = min(i + CHUNK, end)
            out = model(ids[:, i:j], past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(i, j, device="cuda"),
                        logits_to_keep=1)
            if j == end:
                first = int(out.logits[:, -1, :].argmax(dim=-1).item())
    torch.cuda.synchronize()
    return first


def decode_n(dec, first, emb_w, eos):
    gen, cur = [first], first
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(NGEN):
            cur = dec.step(cur, emb_w)
            gen.append(cur)
            if eos is not None and cur == eos:
                break
    torch.cuda.synchronize()
    return gen, time.perf_counter() - t0


def snap_dn(model, cache):
    """48 层 DeltaNet conv/recurrent 快照 (CPU 驻留)。
    注意: cumulative_length 是 attention 层属性, DeltaNet 层无此属性"""
    out = []
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "linear_attention":
            continue
        lay = cache.layers[i]
        out.append((i, lay.conv_states[0].clone().cpu(),
                    lay.recurrent_states[0].clone().cpu()))
    return out


def restore_dn(model, cache, snap):
    for i, conv, rec in snap:
        lay = cache.layers[i]
        lay.conv_states[0].copy_(conv)
        lay.recurrent_states[0].copy_(rec)


def poison_cache(cache):
    """run 间打毒: KV 填 12345.0 + 状态清零 + cumulative_length 归零。
    用有限大值而非 NaN — 规避掩码实现中 NaN×0=NaN 的假阳性歧义;
    restore 任何遗漏 → 值污染 → token 必然发散"""
    for lay in cache.layers:
        # 混合架构: attention 层有 keys/values, DeltaNet 层无该属性 — hasattr 守卫
        if hasattr(lay, "keys") and lay.keys is not None:
            lay.keys.fill_(12345.0)
        if hasattr(lay, "values") and lay.values is not None:
            lay.values.fill_(12345.0)
        if hasattr(lay, "cumulative_length"):
            lay.cumulative_length.zero_()
        cs = getattr(lay, "conv_states", None)
        if isinstance(cs, dict):                       # 部分层 conv_states 为 dict
            for v in cs.values():
                if torch.is_tensor(v) and v.numel():
                    v.fill_(0)
        elif cs is not None and cs.numel():
            cs.fill_(0)
        rs = getattr(lay, "recurrent_states", None)
        if isinstance(rs, dict):                      # recurrent_states 同为 dict
            for v in rs.values():
                if torch.is_tensor(v) and v.numel():
                    v.fill_(0)
        elif rs is not None and rs.numel():
            rs.fill_(0)


def main():
    print("[0/5] 建骨架 + NVFP4 权重上传 ...", flush=True)
    t0 = time.time()
    tok, model = build_model()
    print(f"  构建 {time.time()-t0:.0f}s", flush=True)
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu
    hid = emb_w.shape[1]
    eos = tok.eos_token_id

    # --- 数据构造: 前缀 (CHUNK 对齐) + 两个后缀 ---
    pre = tok(PREFIX_TEXT, return_tensors="pt").input_ids
    L_pre = (min(pre.shape[1], L_PRE_TARGET) // CHUNK) * CHUNK
    assert L_pre >= CHUNK, f"前缀文本不足一个 CHUNK ({pre.shape[1]} tok)"
    pre = pre[:, :L_pre].cuda()

    def mk_suf(q):
        t = tok.apply_chat_template([{"role": "user", "content": q}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        return tok(t, return_tensors="pt").input_ids.cuda()

    sufA, sufB = mk_suf(SUFFIX_A), mk_suf(SUFFIX_B)
    fullA = torch.cat([pre, sufA], dim=1)
    fullB = torch.cat([pre, sufB], dim=1)
    LA, LB = fullA.shape[1], fullB.shape[1]
    assert LB + NGEN + 8 <= MAXLEN
    print(f"[数据] 前缀 {L_pre} tok (CHUNK 对齐) | 后缀A {sufA.shape[1]} tok | "
          f"后缀B {sufB.shape[1]} tok", flush=True)

    # === Run1: baseline A (全量 prefill + graph 捕获 = 生产路径) ===
    print("[Run1] baseline A: 全量 prefill + graph 捕获 ...", flush=True)
    t0 = time.time()
    firstA = prefill(model, cache, fullA, 0, LA)
    t_fullA = time.time() - t0
    mods = patch_fused_deltanet(model, cache)
    vf_snap = [(m._vf_conv.clone().cpu(), m._vf_S.clone().cpu()) for m in mods]
    kv_snap = snap_decode_state(model, cache, LA)
    dec = GraphDecoder(model, cache, LA, hid)
    restore_decode_state(model, cache, kv_snap, LA)
    for m, (c, s) in zip(mods, vf_snap):
        m._vf_conv.copy_(c)
        m._vf_S.copy_(s)
    dec.reset_pos()
    genA, _ = decode_n(dec, firstA, emb_w, eos)
    print(f"  prefill {LA} tok {t_fullA:.2f}s ({LA/t_fullA:.0f} tok/s) | "
          f"decode {len(genA)-1} tok", flush=True)

    # === Run2: baseline B (全量 prefill, graph 复用) ===
    poison_cache(cache)
    print("[Run2] baseline B: 全量 prefill ...", flush=True)
    t0 = time.time()
    firstB = prefill(model, cache, fullB, 0, LB)
    t_fullB = time.time() - t0
    refresh_deltanet_state(model, cache)
    dec.set_pos(LB)
    genB, _ = decode_n(dec, firstB, emb_w, eos)
    print(f"  prefill {LB} tok {t_fullB:.2f}s ({LB/t_fullB:.0f} tok/s) | "
          f"decode {len(genB)-1} tok", flush=True)

    # === Run3: 冷请求 (prefill 前缀 → 快照 → prefill 后缀A) ===
    poison_cache(cache)
    print("[Run3] 冷请求: 前缀 prefill → 快照 → 后缀A prefill ...", flush=True)
    t0 = time.time()
    prefill(model, cache, pre, 0, L_pre)
    t_pre = time.time() - t0
    ts = time.time()
    pre_kv = snap_decode_state(model, cache, L_pre)
    pre_dn = snap_dn(model, cache)
    t_snap = time.time() - ts
    firstA3 = prefill(model, cache, fullA, L_pre, LA)
    refresh_deltanet_state(model, cache)
    dec.set_pos(LA)
    genA3, _ = decode_n(dec, firstA3, emb_w, eos)
    print(f"  前缀 {t_pre:.2f}s + 快照 {t_snap*1000:.0f}ms + "
          f"后缀A {(LA-L_pre)} tok", flush=True)

    # === Run4: 热请求 (restore 前缀快照 → 仅 prefill 后缀B) ===
    poison_cache(cache)
    print("[Run4] 热请求: restore 快照 → 仅后缀B prefill ...", flush=True)
    t0 = time.time()
    restore_decode_state(model, cache, pre_kv, L_pre)
    restore_dn(model, cache, pre_dn)
    t_restore = time.time() - t0
    t0 = time.time()
    firstB4 = prefill(model, cache, fullB, L_pre, LB)
    t_suffix = time.time() - t0
    refresh_deltanet_state(model, cache)
    dec.set_pos(LB)
    genB4, _ = decode_n(dec, firstB4, emb_w, eos)
    print(f"  restore {t_restore*1000:.0f}ms + 后缀B prefill {t_suffix:.2f}s",
          flush=True)

    # === 判定 (推理无损红线: token 逐位一致) ===
    ok3 = genA3 == genA
    ok4 = genB4 == genB
    n_cmp = min(len(genA3), len(genA))
    print("\n===== P4a 验收 =====", flush=True)
    print(f"Run3 (冷+快照)  vs Run1 (baseline A): "
          f"{'PASS' if ok3 else 'FAIL'} ({n_cmp} token 比较)", flush=True)
    print(f"Run4 (热=复用)   vs Run2 (baseline B): "
          f"{'PASS' if ok4 else 'FAIL'}", flush=True)
    t_cold = t_fullB
    t_warm = t_restore + t_suffix
    print(f"首 token 成本: 冷 {t_cold:.2f}s vs 热 {t_warm:.2f}s "
          f"(节省 {(1-t_warm/t_cold)*100:.0f}%, 快照一次性成本 {t_snap*1000:.0f}ms)",
          flush=True)
    if not ok3:
        print("  Run3 差异首位置:", next((k for k in range(n_cmp)
                                      if genA3[k] != genA[k]), None), flush=True)
    if not ok4:
        n2 = min(len(genB4), len(genB))
        print("  Run4 差异首位置:", next((k for k in range(n2)
                                      if genB4[k] != genB[k]), None), flush=True)
    print("\n[Run4 生成内容预览]", flush=True)
    print(tok.decode(genB4[:64]), flush=True)
    return 0 if (ok3 and ok4) else 1


if __name__ == "__main__":
    sys.exit(main())
