# NVFP4 技术复盘 — 坑点·参数·实验全记录

> 汇编：`NVFP4_V100软件模拟研究全量整理_十阶段完整版.md` 阶段 1-17（2026-08-14 至 2026-08-28）
> 定位：**一页看懂整套系统**——架构/参数速查、68 个发现分类索引、静默 bug 专章、验收规程
> 当前基线：**35.3 tok/s（28.32ms/步），63 token 逐位一致，RAG 14 库挂载闭环**

---

## 一、系统架构总览

```
┌────────────────────────────────────────────────────────────────┐
│ 应用层: RAG 问答 (_qwen38_rag.py)                               │
│   检索(CPU): Qwen3-Embedding-0.6B fp32 → kb_search 14库15.9万向量 │
│   生成(GPU): NVFP4 Qwen3.8-27B, MAXLEN=3072, 32-33 tok/s        │
├────────────────────────────────────────────────────────────────┤
│ 推理层: _qwen38_infer.py                                         │
│   QLinear (497个): NVFP4权重常驻 14.41GB                         │
│     T=1 decode: GEMV v8 (PRMT硬查表, 792GB/s) + 成对融合        │
│     T>1 prefill: v5 TensorCore GEMM (wmma, 33.7-38.1 TF)        │
│   融合 kernel (自研 DLL):                                        │
│     nvfp4_attn_decode.cu  — GQA T=1 两遍online-softmax+σgate    │
│     nvfp4_dn_fused.cu      — DeltaNet conv1d+silu+l2norm+rec     │
│     nvfp4_rmsnorm.cu       — 全站161处 RMSNorm 融合              │
│   CUDA Graph (GraphDecoder): 整步decode单graph replay            │
│   KV 复用 (P4a LRU): 多前缀快照 CPU RAM 166-168MiB/套            │
├────────────────────────────────────────────────────────────────┤
│ 量化层: nvfp4_codec.py (E2M1+E8M0, 符号内嵌nibble 0.5625B/param) │
├────────────────────────────────────────────────────────────────┤
│ 硬件: V100-SXM2-16GB (sm_70, 无cp.async/mma PTX, 峰值900GB/s)    │
└────────────────────────────────────────────────────────────────┘
```

## 二、参数速查表

### 2.1 模型与量化

| 项 | 值 |
|---|---|
| 模型 | Qwen3.8-27B（Gated DeltaNet×48 层 + Gated Attention×16 层，hidden 5120，64 层，262K 原生上下文） |
| NVFP4 格式 | E2M1 4bit（非零码点 {0.5,1,1.5,2,3,4,6}）+ E8M0 块缩放（每 16 元素 1B，2^(byte−127)） |
| 显存密度 | **0.5625 B/param**（实测 0.2812 packed 比值精确贴合理论） |
| 权重总量 | 25.63B 参数 → nvfp4_packed.bin 13.42GB + embed NVFP4 CPU 0.68GB + misc 0.01GB |
| GPU 常驻 | **14.41GB / 16GB**（torch 0.01 + DLL 权重 14.41） |
| scale clamp | E8M0 byte ≤254（255 → exp2f(128) 溢出 inf → GEMM 全 NaN，发现 6/7） |

### 2.2 kernel 与性能

| 项 | 值 |
|---|---|
| GEMV v8 | PRMT 硬件查表反量化，`__launch_bounds__(256,8)`，隔离 792GB/s（大 shape）/ 722GB/s（497 核加权） |
| GEMM v5 | wmma TensorCore 三级流水线 + 位构造反量化（`0x3800+(code<<9)`），33.7-38.1 TF |
| attn decode | grid=24×256thr，两遍 online-softmax，fp32 累加，L 从 cumulative_length 动态读；**stride 绑定 MAXLEN 编译期**（DLL 必须与 cache 尺寸一致） |
| rmsnorm | grid=rows，两级 warp 归约，`w1=1+w` 预计算，chunk view 支持 |
| 成对 GEMV | in_proj_b/a（48 对）+ k/v_proj（16 对）拼 buffer 单 launch，-64 launch/步 |
| decode 路径 | poison_cache → prefill(512/块) → refresh_deltanet_state → set_pos(L) → graph replay |
| **速度优化史** | 6.0 → 21.78(R1+R2) → 25.89(v3.2) → 28.8(P5-B) → 32.2(P5-A) → 33.9(P6) → **35.3(P7) tok/s** |

### 2.3 每步时间分布（28.32ms/步，P7 后）

| 类别 | ms/步 | 占比 | 状态 |
|---|---|---|---|
| GEMV v8（497 核） | ~19.2 | 68% | 隔离极限 19.99 − 融合收益；in-graph 干扰 1.74ms 天花板 |
| DN 融合 | 3.7 | 13% | 未深挖 |
| torch elementwise（RoPE/MLP silu·mul/residual） | ~2.0 | 7% | **T2 目标 ~1.3ms** |
| rmsnorm | 1.0 | 3.5% | 已融合 |
| attn v3.1 | 0.19 | 0.7% | 已闭环（4.9ms→0.19ms，25 倍） |

### 2.4 容量与运维

| 项 | 值 |
|---|---|
| OOM 阈值 | MAXLEN ∈ [8192, 9216)——8192 PASS（smi 峰值 16233MiB），9216 差 25MiB |
| 生产推荐 | MAXLEN=6144（留 RAG 余量）；纯推理可 8192 |
| LRU 前缀 | 166-168 MiB/套（CPU RAM），64GB RAM 数百套；HIT restore 70ms（省 95%） |
| V100 掉卡防线 | 关快速启动 + 开机计划任务 GPU-Boot-Repair（自动 remove+rescan）+ TeslaTool 运行中自愈；BIOS Above 4G Decoding 必须 Enabled |
| RAG | 检索 CPU fp32（~2.4GB RAM，热查 <1s）+ 生成 GPU；端到端单题 ~5s |

## 三、68 个发现分类索引

### 3.1 数值精度类（红线区，量化推理的核心约束）

| # | 发现 | 一句话 |
|---|---|---|
| 1 | E2M1 固有 20-30% 相对误差 | 码点稀疏数学特性，非 bug |
| 3 | cosine>0.99 = 方向保持可信 | 量化验收标准 |
| 6 | scale_byte=255 → inf → 全 NaN | 编码期 clamp ≤254 |
| 8-10 | 累加器必须 float32 | FP16 长链累加持续舍入；GEMV 用 half2 链+每 32 元素 float 结算 |
| 44-46 | 视觉塔 NVFP4 FAIL（cos 0.9525） | 浅层敏感+27 层累积；M1 混合精度定案；全局 cos 掩盖行级损伤——双指标验收 |
| **64** | **exp2f vs expf ulp 差异经 63 步贪心解码链式放大** | ulp 级无损论证只对单次前向成立；多步自回归必须按完整 token 链验收 |

### 3.2 性能优化类

| # | 发现 |
|---|---|
| 11-14 | 反量化开销仅 +4.7%（被计算掩盖）；Codec v2 符号内嵌 0.2812 精确贴合；朴素内核四瓶颈 |
| 18-19 | v3 寄存器分块起 NVFP4 反超 FP16（交叉点 T≈1024）；22B 权重 5.8GB 装 V100 |
| 26-29 | v5 TC 正确结构 = 三级流水+位构造；epilogue 对齐；launch_bounds 反噬 |
| 30-31 | Blackwell 差距 236× = 72× 硬件 × 3.28× 软件；数值语义完全等价可外推 |
| 32 | GEMV 专用 kernel decode 28-35× |
| 40 | v3 LUT 419-518GB/s；47: LDS bank conflict 43% 是 LUT 天花板 |
| 48 | PRMT 硬件查表定案（v8，selector 低 16 位陷阱） |
| 49 | V100 无 cp.async 下加深 LDG 流水仍有收益（预设被推翻） |
| 52-53 | decode 97% GPU-bound；非 GEMV GPU 工作 21.6ms 是主战场 |
| 66 | 792GB/s 只是大 shape 瞬时值；尺寸加权真极限 722；小 shape 严重欠占用（48×5120 仅 18GB/s） |

### 3.3 架构机制类

| # | 发现 |
|---|---|
| 35 | cudaDeviceSynchronize 逐调用是吞吐杀手 → 异步发射 |
| 36 | decode 真瓶颈是 CPU dispatch（~2600 次 dtype 转换/步） |
| 37 | Qwen3_5 部署三坑（rope dict/meta to_empty/键映射） |
| 38-39 | R1 潜力>R2；graph 快照必须在 prefill 后立即抓；warmup 污染必须 restore |
| 50-51 | 前缀复用状态集 = KV+conv/recurrent（无 cumulative_length）；收益随前缀线性增长 |
| 57-58 | StaticLayer.update() 用自身 cumulative_length 定位（graph 内 add_(1) 自动推进）；重置状态集必须含 cumulative_length；LRU 服务序列必须与容量联算 |

### 3.4 工程环境类

| # | 发现 |
|---|---|
| 14 | Windows CUDA 编译四坑（BOM/头文件/dllexport/管道截断） |
| 15 | 编辑工具写盘同步延迟 → 原子读写编译流程 |
| 34 | pip 重装必查 ~xxx 残留目录 |
| 55 | NCU 提权 + 英文路径 + `--graph-profiling=node` |
| 59 | `__shfl_down_sync` 只在 warp 内——跨 warp 必须两级（否则 softmax 分母错 8 倍，decode 全 0） |
| 60 | Windows nvcc -shared 不自动导出 C 符号——必须 `__declspec(dllexport)` |
| 61 | CUDA 静默装组件会掏空已有 toolkit 的 include |
| 62 | graph 内 kernel patch 必须 T>1 回退（否则 650MB 缓冲驻留 OOM） |
| 63 | MSVC 代码页 936 误读 UTF-8 注释（C4819）——.cu 统一 utf-8-sig |
| 65 | NCU CSV 步切分 bug（末步截断 ÷3 低估 1.5 倍）——先核验 v8 总数 mod 497 |
| 67 | OOM 阈值前先撞 kernel 编译期 MAXLEN 上限（s_sc 越界 sticky error 误导定位） |
| **68** | **RAG 三连环坑：patch 时序/chat template/MAXLEN stride**（详见第四节） |

## 四、静默 Bug 专章——"不崩溃，只烂输出"

> 本项目最危险的三类 bug：**系统不报错、指标全绿、输出却是垃圾**。复盘共 3 例，全部踩过、修过、已沉淀验收规程。

### 4.1 案例一：exp2f ulp 放大（发现 64，阶段 14）

- **症状**：63 token 验收 FAIL——首 8 token 正确，后续发散。性能指标完全正常。
- **根因**：`exp2f((s·scale·log2e)−m)` 与 `expf(s·scale−m)` 存在 ulp 级差异。单次前向内 softmax 分子分母同源抵消，理论上无害；但**贪心解码是多步链式过程，每步的 argmax 临界 token 会放大任意 ulp 差异**。
- **修复**：v3.1 回退 expf，保留位安全的 half2 加载 + p 预计算。
- **规程**：位精确验收必须按**完整 63 token 链**执行，单点对齐不算数。

### 4.2 案例二：跨 warp 归约丢失（发现 59，阶段 13）

- **症状**：decode 输出全部 token=0，**首 token 正确**（来自 prefill 原路径，极具迷惑性）。
- **根因**：`__shfl_down_sync` 的 width 默认 32，delta>31 是 no-op——直接对 256 线程做 o=128..1 shuffle，只有 warp0 部分和进入结果，softmax 分母错 8 倍。
- **修复**：warp 内 shfl → `s_red[warp]` → warp0 跨 warp 归约 → broadcast。
- **规程**：kernel 正确性先单元测试（已知输入→手算真值），再进管线。

### 4.3 案例三：RAG 三连环（发现 68，阶段 17）

- **症状**：RAG 管线跑通、检索 6/6 命中、tok/s 正常，**回答是"!!!!!"乱码**。
- **三个叠加根因**（逐个隔离修复）：
  1. `patch_fused_deltanet` 在 prefill 前调用 → conv_states=None 崩溃（这个会报错，反而好修）；
  2. 裸文本 prompt 不走 `apply_chat_template` → Qwen3.8 输出退化；
  3. `patch_attn_decode` 硬编码 3072-stride DLL，cache=6144 → KV 寻址错位 → decode 乱码（修完 2 才暴露 3）。
- **规程**：**"管线跑通"≠"语义正确"——性能指标全绿时输出仍可能是乱码，验收必须含人读语义质量项**；多 bug 叠加时每修一个重跑全量验收。

### 4.4 静默 bug 的共同模式与防线

| 模式 | 案例 | 防线 |
|---|---|---|
| 首 token 正确、decode 全烂 | 59/68-3 | prefill 与 decode 路径分开验收 |
| 指标正常、内容乱码 | 68 | 人读语义验收项（不可省） |
| ulp 级差异链式放大 | 64 | 完整 63 token 链逐位对照 |
| warmup/重 capture 污染状态 | 39/P7 | 每次重 capture 后必须 snap/restore |
| sticky error 冒头位置误导 | 67 | CUDA_LAUNCH_BLOCKING=1 复跑定位真凶 |
| 打毒通过掩盖重置遗漏 | 57 | poison_cache 是防回归锚点，重写逐行 diff |

## 五、实验台账总表（阶段 1-17）

| 阶段 | 内容 | 关键数字 | 状态 |
|---|---|---|---|
| 1 | 格式解析 + Python codec | cosine ≥0.9926 | ✅ |
| 2 | 单缓冲 fused dequant-GEMM | 模拟器全对齐误差=0 | ✅ |
| 3 | 双缓冲 2-stage + V100 实测 | 反量化开销 +4.7% | ✅ |
| 4 | DiT benchmark + kernel 五代 v2→v5 | 33.7-38.1 TF（12.8× vs FP16） | ✅ |
| 5 | Blackwell 差距分析 | 236× = 72× 硬件 × 3.28× 软件 | ✅ |
| 6 | Qwen3.8-27B 实机部署 | 14.42GB/16GB，decode 6.0 tok/s | ✅ |
| 7 | R1 vs R2 对决 | 组合 21.78 tok/s（3.77×） | ✅ |
| 8 | GEMV v3 LUT + MTP | 518GB/s；MTP 净亏损定论 | ✅ |
| 8C | GEMV v4→v8 PRMT | 792GB/s（+53%） | ✅ |
| 9 | 8K 上下文（分块 prefill 三层架构） | plen=8090 全链 OK | ✅ |
| 10 | 视觉塔 M1 混合精度 | cos 0.99988，513MB | ✅ |
| 12 | P4a 前缀复用 + P4b profile + LRU | 首 token 省 61%；LRU 6/6；21.6ms 主战场 | ✅ |
| 13 | P5-B attn kernel + P5-A elementwise | 24.8→32.2 tok/s | ✅ |
| 14 | P6 attn v3.1 + NCU 勘误 | 33.9 tok/s；发现 64/65 | ✅ |
| 15 | P7 shape 裁决 + 成对融合 | **35.3 tok/s**（+42%） | ✅ |
| 16 | P8 压测 OOM 阈值 | [8192, 9216)；发现 67 | ✅ |
| 17 | RAG 14 库挂载 | 6/6 PASS；发现 68 | ✅ |

**知识库沉淀**：cezanne.db nvfp4 系列 + kb 系列（052-057 等），总数 11459 条，FTS 全验证。

## 六、验收规程（红线，历次教训固化）

1. **数值正确性**：
   - kernel 单元测试：FP64 真值，验收三件套 mean_rel + p99_rel + max_abs（v8 范例：1.65e-03/2.74e-02/0.0018）；
   - 端到端：**63 token 逐位一致**（贪心链，非单点）；
   - 量化层：全局 cosine ≥0.99 **且** 行最低 ≥0.98（双指标）；
   - kernel 数值误差必须 ≪ 量化误差（mean_rel<0.01 且 max_abs<1% 输出幅值）。
2. **语义质量**：人读输出（防静默乱码——发现 68 教训）。
3. **性能口径**：wall-clock 为准（NCU 只做占比归因）；跨会话引用数字先查库（发现 54）。
4. **graph 相关**：重 capture 后必须 restore；decode-only patch 必须 T>1 回退。
5. **多轮状态**：poison_cache 打毒 → 逐位对照；重置集含 cumulative_length。

## 七、复用清单（下一模型可直接带走）

- 量化层：`nvfp4_codec.py`（编码器 100% 复用——发现 31 数值语义与解码路径无关）
- GEMM 层：v5 TensorCore kernel（sm_70 最优，mma PTX 需 sm_75+）
- GEMV 层：v8 PRMT kernel + 成对融合模式（依赖"同输入成对调用"结构，需按模型重排）
- 融合 kernel：rmsnorm（通用）/attn decode（GQA 结构需适配）/dn_fused（DeltaNet 专属）
- 部署三步：meta 骨架 → patch Linear → to_empty 填充（键映射按模型调整）
- 验收体系：上述全套规程
- 显存公式：权重 NVFP4 0.5625 B/param；16GB 上限 ≈25.6B 纯权重（27B 为工程上限）

---

*文档生成：2026-08-28，基于 17 阶段完整实验记录汇编。性能、精度、容量数字均为实测值。*
