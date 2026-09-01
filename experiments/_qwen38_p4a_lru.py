# -*- coding: utf-8 -*-
"""S2-P4a+: 多前缀 KV 缓存 + LRU 淘汰 (单机版 RadixAttention 第二步)
=====================================================================
升级 P4a: 单前缀 → 多前缀字典 (OrderedDict LRU, 字节容量上限)。
- key   = 前缀 token 序列 SHA256 (token 级, 非文本级)
- value = (KV 快照, DeltaNet 快照) CPU RAM 驻留
- 驱逐  = 超容量时逐出最久未使用
验收: 3 前缀 P1/P2/P3 × 容量=2 → 交错请求强制驱逐+重建, 全部
      与全量 prefill baseline 63 token 逐位一致 + 打毒。
用法: python _qwen38_p4a_lru.py
"""
import hashlib
import sys
import time
from collections import OrderedDict

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

P1_TEXT = ("人工智能发展史概述。人工智能的概念诞生于一九五六年达特茅斯会议, "
           "麦卡锡、明斯基等学者首次提出用机器模拟智能的设想。此后六十年间, "
           "经历了符号主义、连接主义与行为主义三大流派的兴衰交替。"
           "一九八零年代的专家系统曾经带来第一波商业化浪潮, 医疗诊断、化学"
           "分析、地质勘探各领域涌现出大量应用, 但因知识获取瓶颈与维护成本"
           "问题陷入低谷, 人工智能迎来第一次寒冬。同一时期, 反向传播算法的"
           "重新发现为神经网络研究注入活力, 多层感知机重新受到关注。"
           "一九九零年代支持向量机以严密的统计学习理论基础成为主流, 核方法"
           "与结构风险最小化原则大放异彩。二零一二年 AlexNet 在 ImageNet "
           "竞赛中以显著优势夺魁, 深度学习复兴的大幕正式拉开, 卷积网络在"
           "计算机视觉领域攻城略地, 图像分类、目标检测、语义分割纪录被不断"
           "刷新。循环网络与长短期记忆网络则统治了自然语言处理领域, 机器"
           "翻译、语言模型、序列标注全面转向神经方法。二零一七年 Transformer "
           "架构横空出世, 自注意力机制抛弃了循环结构, 完全基于注意力实现"
           "序列建模, 可并行性带来前所未有的训练效率。此后预训练范式确立, "
           "大规模无监督语料上学习通用表示, 再下游微调成为标准流程。二零"
           "二零年 GPT-3 展示了大规模语言模型的涌现能力, 上下文学习、"
           "思维链推理等能力随规模自然涌现, 缩放定律成为行业共识。近年来"
           "的研究方向包括检索增强生成、多模态融合、智能体架构、工具调用"
           "以及对齐技术。开源社区贡献了 LLaMA、Mistral 等高质量基座, "
           "推动了整个生态的繁荣。推理优化同样是研究热点, 量化、蒸馏、"
           "投机解码、缓存复用等技术在降低部署成本方面效果显著。未来的"
           "发展方向包括更高效的架构设计、更强的推理能力、更可控的生成"
           "质量, 以及与物理世界更深入的交互。人工智能的历史就是一部"
           "范式更迭史, 每次转折都源于关键思想与算力基础的共振。展望未来, "
           "神经符号融合试图把学习与推理的优势结合起来, 世界模型让智能体"
           "在内部模拟环境后果, 具身智能把感知决策闭环延伸到物理世界。安全"
           "与对齐研究确保能力增长不脱离人类意图, 可解释性努力打开黑箱。"
           "算力基础设施从通用图形处理器走向专用推理芯片, 互连带宽与存储"
           "层次成为新瓶颈。开源与闭源路线并行发展, 竞争推动能力普惠。"
           "数据质量、合成数据与课程学习成为新的研究焦点, 强化学习从游戏"
           "走向代码与数学等可验证领域, 过程监督比结果监督更有效。多智能"
           "体协作展示了分工与辩论的价值。人工智能的每一次进步都伴随着对"
           "智能本质的更深理解, 这场探索远未终结。")
P2_TEXT = ("宏观经济学核心理论速览。宏观经济学研究整体经济的运行规律, 核心变量"
           "包括国内生产总值、通货膨胀率、失业率与国际收支。古典学派相信市场的"
           "自我调节能力, 萨伊定律认为供给会自动创造需求, 价格机制会引导资源"
           "自动流向最有价值的用途。大萧条动摇了古典理论的根基, 凯恩斯革命"
           "应运而生, 指出有效需求不足可能导致长期萧条, 工资刚性使得劳动市场"
           "无法自动出清, 政府应当运用财政政策与货币政策平滑经济周期。IS-LM "
           "模型将产品市场与货币市场的一般均衡形式化, 成为短期分析的基石。"
           "菲利普斯曲线揭示了失业与通胀之间的替代关系, 为政策制定提供了"
           "菜单。货币主义代表人物弗里德曼强调货币供应量对物价水平的决定性"
           "作用, 通货膨胀归根结底是一种货币现象, 自然失业率假说指出长期"
           "菲利普斯曲线是垂直的, 需求管理政策无法永久压低失业率。理性预期"
           "学派进一步指出, 经济主体会利用一切可用信息形成预期, 系统性的"
           "政策干预可能被预期抵消, 卢卡斯批判从根本上质疑了传统计量模型"
           "的政策评估效力。真实经济周期理论把技术冲击视为波动源泉, 主张"
           "市场即使在短期也是出清的。新凯恩斯主义为价格粘性提供了微观基础, "
           "菜单成本、交错定价等模型解释了名义刚性, 动态随机一般均衡框架"
           "成为当代央行政策分析的标准工具, 泰勒规则描述了利率对通胀缺口"
           "与产出缺口的系统性反应。内生增长理论把技术进步内生化, 人力资本、"
           "研发投入、知识外溢解释了长期增长的源泉与规模效应。发展经济学"
           "关注贫困陷阱、制度质量与文化因素对增长路径的塑造。国际贸易"
           "理论从比较优势出发, 新贸易理论引入规模经济与产品差异化, 解释"
           "了产业内贸易现象。宏观经济学的历史就是政府与市场关系的再平衡史, "
           "每一次危机都催生新的理论范式, 而旧智慧也总在新的形式下复活。"
           "宏观政策实践同样在不断演进。通胀目标制赋予央行独立性, 以可信"
           "承诺锚定预期。量化宽松在利率触及零下界后打开非常规工具箱, 资产"
           "负债表成为新政策变量。前瞻指引通过沟通管理市场预期。财政货币"
           "政策的边界、主权债务的可持续性、人口老龄化对潜在增速的拖累、"
           "全球化退潮与供应链重构对通胀动态的影响, 都是当代宏观研究的"
           "前沿议题。收入分配与包容性增长提醒我们, 效率之外还有公平维度。"
           "金融稳定成为新的政策目标, 宏观审慎工具试图驯服信贷周期。经济"
           "计量的识别策略从结构模型走向因果推断, 双重差分与合成控制法"
           "提高了经验研究的可信度。大数据与机器学习为宏观经济测量提供"
           "了新手段, 卫星夜光、高频文本情绪等另类数据补充了传统统计。")
P3_TEXT = ("软件工程方法论纲要。软件工程诞生于一九六八年北约软件工程会议, 目的"
           "是应对软件危机, 当时的项目普遍超出预算、延期交付、质量失控。结构化"
           "方法强调自顶向下逐步求精, 数据流图、结构图与模块化是其核心工具, "
           "信息隐藏原则奠定了模块设计的基础。面向对象方法以封装、继承、多态"
           "为支柱, 统一建模语言成为标准表达, 设计原则后来被提炼为 SOLID 五"
           "大原则。设计模式运动整理了二十三种经典解法, 工厂、单例、观察者、"
           "策略、模板方法成为交流的通用词汇。组件化与中间件时代, 企业级开发"
           "以 CORBA、COM、EJB 为代表, 分布式对象技术初现端倪。敏捷宣言提出"
           "个体互动高于流程工具, 可工作的软件高于详尽的文档, 客户合作高于"
           "合同谈判, 响应变化高于遵循计划。极限编程的实践包括结对编程、测试"
           "驱动开发、持续集成与重构, Scrum 以冲刺、每日站会、回顾会议组织"
           "迭代节奏, 看板方法通过可视化流水线限制在制品数量暴露瓶颈。"
           "领域驱动设计强调统一语言与限界上下文, 聚合根、实体、值对象、"
           "仓储与工厂构成模型词汇, 战略设计把大系统划为多个有界上下文。"
           "微服务架构按业务能力拆分部署单元, 每个服务独立演进独立伸缩, "
           "配合容器化、服务网格与持续交付成为云时代主流, 但也带来了分布式"
           "事务、服务发现、链路追踪等新复杂度。DevOps 打破开发与运维的墙, "
           "建立自动化流水线, 基础设施即代码让环境变更可审查可回滚。可观测"
           "性三大支柱是指标、日志与追踪。测试金字塔提倡大量单元测试、适量"
           "集成测试、少量端到端测试。混沌工程主动注入故障验证系统韧性。"
           "软件工程的历史就是与复杂度做斗争的历史, 每一代方法论都在回答"
           "同一个问题: 如何让一群普通人可靠地构建超出个体理解能力的系统。"
           "当代工程实践继续分化演进。函数式编程思想回流主流, 不可变性"
           "与纯函数简化并发推理。类型系统从泛型走向依赖类型的表达力阶梯。"
           "响应式架构以事件流解耦组件, 背压机制保护下游不被压垮。数据"
           "工程与机器学习工程引入新的生命周期, 特征平台、模型注册中心、"
           "线上评估与影子部署构成生产级机器学习管道。数据质量、漂移监测"
           "与回滚策略决定模型上线后的命运。性能工程关注延迟与吞吐的权衡, "
           "缓存层次、零拷贝、批处理与异步是四大杠杆。可靠性工程以服务"
           "级别目标量化用户体验, 熔断、限流与降级守护核心链路。安全"
           "左移把威胁建模融入设计阶段, 供应链安全成为新战场。架构决策"
           "记录让重大取舍有据可查, 架构适应度函数守护演进不偏离意图。"
           "工程文化最终决定方法论的落地成色, 无指责事后复盘与持续学习"
           "是高绩效组织的共同特征。工具永远在变, 围绕反馈环缩短与认知"
           "负荷降低的原则不变。")

SUFFIX = "请用两句话概括以上内容的要点。"


def prefill(model, cache, ids, start, end):
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
    with torch.no_grad():
        for _ in range(NGEN):
            cur = dec.step(cur, emb_w)
            gen.append(cur)
            if eos is not None and cur == eos:
                break
    torch.cuda.synchronize()
    return gen


def snap_dn(model, cache):
    out = []
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "linear_attention":
            continue
        lay = cache.layers[i]
        out.append((i, lay.conv_states[0].clone().cpu(),
                    lay.recurrent_states[0].clone().cpu()))
    return out


def restore_dn(cache, snap):
    for i, conv, rec in snap:
        lay = cache.layers[i]
        lay.conv_states[0].copy_(conv)
        lay.recurrent_states[0].copy_(rec)


def poison_cache(cache):
    for lay in cache.layers:
        if hasattr(lay, "keys") and lay.keys is not None:
            lay.keys.fill_(12345.0)
        if hasattr(lay, "values") and lay.values is not None:
            lay.values.fill_(12345.0)
        # StaticLayer.update() 忽略传入的 cache_position, 用 cumulative_length 定位
        # KV 写入槽位; graph decode 内 add_(1) 被 capture, 每次 replay 自动 +1。
        # 不归零 → 下次 prefill 从残留位置写入 → 槽位错乱 + 越界断言 (LRU FAIL 根因)
        if hasattr(lay, "cumulative_length"):
            lay.cumulative_length.zero_()
        cs = getattr(lay, "conv_states", None)
        if isinstance(cs, dict):
            for v in cs.values():
                if torch.is_tensor(v) and v.numel():
                    v.fill_(0)
        elif cs is not None and cs.numel():
            cs.fill_(0)
        rs = getattr(lay, "recurrent_states", None)
        if isinstance(rs, dict):
            for v in rs.values():
                if torch.is_tensor(v) and v.numel():
                    v.fill_(0)
        elif rs is not None and rs.numel():
            rs.fill_(0)


class PrefixCache:
    """多前缀 KV+DeltaNet 快照 LRU 缓存 (CPU RAM, 字节容量上限)"""

    def __init__(self, model, cache, max_bytes):
        self.model, self.cache = model, cache
        self.max_bytes = max_bytes
        self.store = OrderedDict()          # sha16 -> (kv_snap, dn_snap, L)
        self.hit = self.miss = self.evict = 0

    @staticmethod
    def key_of(ids):
        return hashlib.sha256(ids.cpu().numpy().tobytes()).hexdigest()[:16]

    @staticmethod
    def _bytes(kv, dn):
        b = sum(t[1].numel() * t[1].element_size() for t in kv)
        b += sum(t[1].numel() * t[1].element_size() +
                 t[2].numel() * t[2].element_size() for t in dn)
        return b

    def put(self, key, kv, dn):
        self.store[key] = (kv, dn)
        self.store.move_to_end(key)
        while (len(self.store) > 1 and
               sum(self._bytes(v[0], v[1]) for v in self.store.values())
               > self.max_bytes):
            victim, _ = self.store.popitem(last=False)
            self.evict += 1
            print(f"    [LRU] 驱逐 {victim}", flush=True)

    def get(self, key):
        if key in self.store:
            self.hit += 1
            self.store.move_to_end(key)
            return self.store[key]
        self.miss += 1
        return None


def main():
    print("[0] 建骨架 + NVFP4 权重上传 ...", flush=True)
    tok, model = build_model()
    cache = StaticCache(config=model.config, max_cache_len=MAXLEN)
    ensure_kv_stage(model, MAXLEN)
    emb_w = model.model.embed_tokens.weight_cpu
    hid = emb_w.shape[1]
    eos = tok.eos_token_id

    # 3 套前缀 (各自 CHUNK 对齐), 公共后缀模板
    pres = []
    for txt in (P1_TEXT, P2_TEXT, P3_TEXT):
        ids = tok(txt, return_tensors="pt").input_ids
        L = (ids.shape[1] // CHUNK) * CHUNK
        assert L >= CHUNK
        pres.append(ids[:, :L].cuda())

    def mk_suf():
        t = tok.apply_chat_template([{"role": "user", "content": SUFFIX}],
                                    tokenize=False, add_generation_prompt=True,
                                    enable_thinking=False)
        return tok(t, return_tensors="pt").input_ids.cuda()

    suf = mk_suf()
    fulls = [torch.cat([p, suf], dim=1) for p in pres]
    Ls = [f.shape[1] for f in fulls]
    print(f"[数据] 前缀 {[p.shape[1] for p in pres]} tok | 后缀 {suf.shape[1]} tok",
          flush=True)

    # === Run0: baseline 全量 prefill ×3 (P1/P2/P3) ===
    baselines = []
    for k, f in enumerate(fulls):
        poison_cache(cache)
        first = prefill(model, cache, f, 0, Ls[k])
        if k == 0:
            mods = patch_fused_deltanet(model, cache)
            vf_snap = [(m._vf_conv.clone().cpu(), m._vf_S.clone().cpu())
                       for m in mods]
            kv_snap = snap_decode_state(model, cache, Ls[0])
            dec = GraphDecoder(model, cache, Ls[0], hid)
            restore_decode_state(model, cache, kv_snap, Ls[0])
            for m, (c, s) in zip(mods, vf_snap):
                m._vf_conv.copy_(c)
                m._vf_S.copy_(s)
            dec.reset_pos()
        else:
            refresh_deltanet_state(model, cache)
            dec.set_pos(Ls[k])
        baselines.append(decode_n(dec, first, emb_w, eos))
        print(f"  [baseline] P{k+1} 全量 prefill {Ls[k]} tok 完成", flush=True)

    # 容量设为 ~2 个前缀快照的大小 (从 P1 快照实测字节数推)
    _, kv1 = None, None
    poison_cache(cache)
    prefill(model, cache, pres[0], 0, pres[0].shape[1])
    kv1 = snap_decode_state(model, cache, pres[0].shape[1])
    dn1 = snap_dn(model, cache)
    snap_bytes = PrefixCache._bytes(kv1, dn1)
    cap = snap_bytes * 2 + 1        # 只放得下 2 个 → 第 3 个进来必驱逐最旧
    pc = PrefixCache(model, cache, cap)
    print(f"\n[LRU] 快照 {snap_bytes/1e6:.0f}MB/前缀, 容量上限 "
          f"{cap/1e6:.0f}MB (≈2 前缀)", flush=True)

    def serve(k):
        """一次完整请求: 查缓存 → 命中 restore / 未命中 prefill+快照 → prefill 后缀"""
        p, f, L = pres[k], fulls[k], Ls[k]
        key = PrefixCache.key_of(p)
        t0 = time.time()
        hit = pc.get(key)
        if hit is not None:
            kv, dn = hit
            poison_cache(cache)
            restore_decode_state(model, cache, kv, p.shape[1])
            restore_dn(cache, dn)
            t_rest = time.time() - t0
            tag = "HIT "
        else:
            poison_cache(cache)
            prefill(model, cache, p, 0, p.shape[1])
            kv, dn = snap_decode_state(model, cache, p.shape[1]), \
                snap_dn(model, cache)
            pc.put(key, kv, dn)
            t_rest = time.time() - t0
            tag = "MISS"
        first = prefill(model, cache, f, p.shape[1], L)
        refresh_deltanet_state(model, cache)
        dec.set_pos(L)
        gen = decode_n(dec, first, emb_w, eos)
        return gen, tag, t_rest

    # === 服务序列: P1(存) P2(存) P3(驱逐P1) P2(hit) P3(hit) P1(驱逐P2重建) ===
    seq = [0, 1, 2, 1, 2, 0]
    results = []
    for k in seq:
        gen, tag, t = serve(k)
        ok = gen == baselines[k]
        results.append(ok)
        print(f"  [serve] P{k+1} {tag} | pre_restore {t*1000:.0f}ms | "
              f"vs baseline {'PASS' if ok else 'FAIL'}", flush=True)

    print(f"\n===== S2 验收 =====")
    print(f"逐位一致: {sum(results)}/{len(results)} "
          f"{'全部 PASS' if all(results) else '存在 FAIL'}")
    print(f"LRU 统计: hit={pc.hit} miss={pc.miss} evict={pc.evict} "
          f"驻留={len(pc.store)}")
    return 0 if all(results) and pc.evict >= 1 and pc.hit >= 1 else 1


if __name__ == "__main__":
    sys.exit(main())
