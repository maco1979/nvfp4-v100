# NVFP4 V100 软件模拟研究全量整理（10 阶段完整版）

> 汇编基准：`NVFP4_研究记忆.md` 阶段 1-10 全部实测数据（2026-08-14 至 2026-08-17）
> 本版修正：阶段 4/5 状态（均已完成）、"性能弱于 FP16"过时结论（阶段 4 起已反超）、项目定位（benchmark → 27B LLM 实机部署）

---

## 一、项目基础信息

### 1.1 研究目标（随进展三次扩展）

1. **原始目标**：完整解析 Blackwell NVFP4（E2M1 4bit 浮点 + 每 16 元素 E8M0 块缩放）格式；V100 手搭全链路软件模拟；精度/速度实测；对比硬件原生差距。验证载体为 DiT 视频模型 benchmark。
2. **扩展一（阶段 6）**：用 NVFP4（0.5625 B/param）把 Qwen3.8-27B（27B Dense，Gated DeltaNet + Gated Attention 混合，hidden 5120，64 层，262K 原生上下文）塞进 V100 16GB 实机推理——FP16 全量 54GB 双超限（RAM 32GB / VRAM 16GB），NVFP4 常驻 GPU 是唯一路径。
3. **扩展二（阶段 10/P1）**：视觉塔（27 层 Transformer，110 Linear）单独量化验证，为多模态铺路。

- 开发路线：自研全链路（路线 A），不复用 NF4 库，完整吃透 bit 布局、block-scale、融合反量化 GEMM；
- 测试原则：小矩阵单元测试先行 → 真实尺寸回放 → 整塔端到端 → 实机部署。

### 1.2 知识库支撑（实测修正）

- **cezanne 库**：本研究主力支撑，nvfp4 系列 30 条入库（nvfp4-001..029 + 021r），六类分布：cuda_optimization 8 / error_isolation 7 / code_patterns 5 / benchmark_data 5 / nvfp4_architecture 4 / process_rule 1；
- **davinci/galileo 库**：早期 TensorCore 加速范式、矩阵基础、复杂度基线条目（阶段 5 差距分析引用 tensorcore_generation/precision/sparse/misconception 条目）；
- NVFP4 为新硬件格式无存量条目，采用模型原生知识 + CPHYSJEPA 物理一致性校验（码点单调性、E8M0 为 2 的幂、编解码确定性）。

### 1.3 十阶段整体规划（最终状态）

| 阶段 | 工作内容 | 最终状态 |
|---|---|---|
| 1 | NVFP4 格式解析、Python 编解码参考实现、小矩阵验证 | ✅ 完成 |
| 2 | 单缓冲融合反量化-GEMM CUDA 内核开发与逻辑验证 | ✅ 完成 |
| 3 | V100 双缓冲 2-stage 内核、FP16 基线对照、GPU 实测 + Codec v2 | ✅ 完成 |
| 4 | DiT 视频模型完整 benchmark（LTX-2.3 22B 真实尺寸回放 + kernel 五代优化 v2→v5） | ✅ 完成（v5 TensorCore 12.8× vs FP16） |
| 5 | V100 软件模拟 vs Blackwell 硬件 NVFP4 差距分析报告 | ✅ 完成（236× = 72× 硬件 × 3.28× 软件） |
| 6 | Qwen3.8-27B 实机部署（NVFP4 常驻 16GB V100） | ✅ 完成（14.42GB/16GB，跑通生成） |
| 7 | R1(CUDA Graph) vs R2(fused kernel) 优化路线对决 | ✅ 完成（R1+R2 组合 21.78 tok/s，63/63 一致） |
| 8 | GEMV 带宽优化 v3 LUT + MTP 投机解码对决 | ✅ 完成（518 GB/s；MTP 净亏损定论） |
| 9 | 8K 上下文跑通（分块 prefill 笔记式架构） | ✅ 完成（plen=8090 全链 OK） |
| 10 | 视觉塔量化 P1（多模态前置验证） | ✅ 完成（M1 混合精度定案 cos=0.99988） |

**当前最佳推理基线：R1+R2+GEMV v3.2 = 25.89 tok/s（63/63 + 257/257 token 一致）**

---

## 二、分阶段详细成果、关键发现、工程问题与修复

### 阶段 1：格式研究与 Python 参考实现

交付物：`nvfp4_codec.py`（NVFP4 标准编解码基准）。

**NVFP4 规格**：E2M1 4bit（bit3 符号 / bit2-1 指数 bias=1 / bit0 尾数；非零码点 {0.5,1,1.5,2,3,4,6}，间距不均匀）+ E8M0 块缩放（每 16 元素 1 字节，值=2^(byte-127)，区间 [2^-127,2^128]）；最终数值 = E2M1 × scale。6 项测试全 PASS，校准矩阵 cosine ≥0.9926。

**发现 1-4**：(1) 随机数据 20-30% 相对误差是 E2M1 码点稀疏的固有数学特性，非 bug；(2) 量化评估用 NRMSE/cosine/SNR，相对误差对趋零值无意义；(3) cosine>0.99 即向量方向保持、推理可信；(4) 简单方差缩放校准无效——零附近仅 {0,0.5} 两码点，高精度需 GPTQ/AWQ 式权重重排。

**编码规范**：反量化中间计算 float64 防 scale 溢出；双元素打包 uint8；scale 指数限 [-127,127]（发现 6 修复后）。

### 阶段 2：单缓冲 Fused dequant-GEMM 内核

环境阻塞期（V100 驱动损坏 + 无 MSVC）用纯 Python 模拟器逐元素复刻 CUDA 逻辑完成前置验证。除 scale_byte=255 边界外全部对齐误差=0。

**发现 5-7**：(5) 两套内核算法逻辑完全正确；(6) `exp2f` 在 scale_byte=255 溢出为 inf → GEMM 全 NaN，修复：scale_exp 上限 128→127，保证 2^127 在 float32 内；(7) Python 模拟器是高效无 GPU 验证路径。

### 阶段 3：双缓冲 2-stage 内核 + V100 实测 + Codec v2

**发现 8-10**：双缓冲只改访存时序不改计算结果；模拟器不可简化加载逻辑/不可提前截断 FP16 累加；GEMM 累加器必须 float32（分段 FP16 累加持续舍入）。V100 无 cp.async，靠 `__syncthreads()` 伪重叠。

**环境修复全流程**：V100 Tesla 驱动手动恢复 + TCC 模式；MSVC Build Tools winget 安装；CUDA 11.8 bin 丢失 → 12.8.1 离线包手动自定义安装（CUDA 13 不支持 sm_70，pip 包无完整 nvcc）。

**GPU 实测**：真值 3/3 PASS（1-2 ULP 属 FP16 正常偏差）。

**发现 11-16**：(11) 反量化实测开销仅 +6.6%，推翻 30-45% 理论预估——朴素内核利用率 8%，瓶颈掩盖指令开销；(12) 独立 signs 数组致显存比 0.7812 → Codec v2 符号内嵌 nibble（Blackwell 布局）→ **0.2812 精确贴合理论 0.28125**，反量化开销再降至 +4.7%；(13) 朴素内核四大瓶颈：无寄存器分块/无向量化加载/逐元素转换/无 TensorCore；(14) Windows CUDA 编译四坑：UTF8 无 BOM 中文注释、缺 cuda_fp16.h、DLL 导出缺 dllexport、PowerShell 管道截断进程；(15) 编辑工具写盘同步延迟——必须落盘的改动用原子读写编译流程。

**D3 补注（2026-08-17，发现 49）**：3-stage 缓冲对照（同源 `nvfp4_cuda_v2.cu` 以 `-DNUM_STAGES` 编译，SMEM 8→12KB/block）——256³ 无收益（+1.3% 慢，prologue 占比高），512³/1024³/512×512×5120 收益 +16.6%/+11.8%/+19.4%，**平均 +14.1%，bit-exact PASS**。**发现 49：V100 无 cp.async 下加深 LDG 流水（提前 2 tile 发起全局读）仍有实质收益，"无 cp.async 则多 stage 无用"的预设被推翻**；大 K（5120，k_tiles=160）场景收益最大。后续 v5 TensorCore 内核值得复验同款 stage 加深。

### 阶段 4：DiT 完整 benchmark + kernel 五代优化（v2→v5）

LTX-2.3 22B 尺寸分布提取（1777 个 2D Linear，22.15B 参数，前 8 种尺寸覆盖 97.1% 计算开销；**发现 17**：双流 MM-DiT 结构，主流 4096 + 次级流 2048）。36 组合真实尺寸回放（T∈{512,1536,4992,14080}）。

| 指标 | v2 朴素 | v3 寄存器分块 | v4 __hfma2 | v5 TensorCore wmma |
|---|---|---|---|---|
| 大 T 吞吐平台 | 2.88 TF | 11.0-11.9 TF | 18.5-20.1 TF | **33.7-38.1 TF** |
| 加权前向 T=14080 | 210.2s | 54.0s | 32.3s | **17.1s** |
| NVFP4/FP16 时间比 | 0.94-1.08 | 0.23-0.37 | 0.137-0.194 | **0.074-0.116** |
| 对上一代加速 | — | 3.89× | 1.67× | 1.93× |

- **发现 18（推翻"性能弱于 FP16"）**：v3 寄存器分块起 NVFP4 全部 36 组合反超 FP16——计算/访存比 ×4 后反量化被计算掩盖；交叉点 T≈1024，带宽节省开始回报。22B 权重 41.3GB → **5.8GB（1:7.11）**，可整体装入 V100 16GB（**发现 19**）。
- **发现 20**：回放测速方法论——`python -u` 防管道缓冲假死；7.7 亿元素权重分块编码防峰值内存爆。
- **发现 21-25**（v3/v4）：寄存器分块消除反量化显性化；v4 后瓶颈转移至指令吞吐；两级累加精度代价 → 验收标准方法论"**kernel 数值误差 ≪ 量化误差**"（mean_rel<0.01 且 max_abs<max(0.1, 1%×输出幅值)）；MSVC GBK 编码坑（.cu 必须 UTF-8 带 BOM）。
- **发现 26-29**（v5）：TensorCore 融合反量化正确结构 = 三级流水线 + 位构造反量化（`0x3800+(code<<9)` 替代查表，~3 条/元素）；epilogue 打包写对齐陷阱（NaN 假象）；`__launch_bounds__` 强推 occupancy 反噬（spill 回退）；长基准早期停滞判定标准（CPU 停滞 + GPU 0% + 无输出 → 杀重跑）。

### 阶段 5：Blackwell 差距分析报告

交付物：`fp4研究\NVFP4_阶段5_Blackwell差距分析.md`。

| 口径 | 差距 |
|---|---|
| 峰值对峰值 | 236×（B200 FP4 TC 9,000 TF ÷ v5 实测 38.08 TF） |
| 实测对实测 [估] | ~140×（B200 按 60% GEMM 效率推算） |
| 格式数值语义 | **1×（完全等价）** |
| 显存带宽 | 8.9× |

- **发现 30**：差距乘法分解 236× = 72× 硬件代际（FP4 vs FP16 通路 4× × 微架构 18×）× 3.28× 软件损耗（=1÷30.5% TC 利用率）。
- **发现 31**：**数值语义完全等价**——RTN 量化误差在编码侧决定、与解码路径无关，cosine 0.9926+/SNR 18.3dB 结论可直接外推 Blackwell；编码器/验收标准/测试方法论 100% 复用，kernel 换 CUTLASS SM100 block-scaled GEMM。

### 阶段 6：Qwen3.8-27B 实机部署（27B 塞进 16GB V100）

量化产物：`nvfp4_packed.bin` 13.42GB（25.63B 参数 × 0.5625 B/param）+ fp16 embed 2.37GB（CPU 常驻）+ misc 0.01GB，index 497 张量与骨架 497 Linear 完全吻合。

| 指标 | 数值 |
|---|---|
| 权重显存 | **14.42GB / 16GB** |
| prefill | 16-21 tok/s（v5 TensorCore GEMM 路径） |
| decode（初版） | 6.0-6.1 tok/s（GEMV 路径，63 tok 稳定） |
| 生成质量 | 连贯、事实正确、自识别正确 |

- **发现 32**：GEMV 专用 kernel（v1 warp/行 + E8M0 位构造 + warp shuffle 规约）decode 场景 28-35× 提升，实测 305-392 GB/s。
- **发现 33**：E2M1 位技巧编码跨 binade 陷阱——{1.0,1.5} 与 {2.0,3.0} 跨界舍入使位技巧 FAIL，保持 searchsorted 正确性优先（CPU encode 13M elem/s 非关键路径）。
- **发现 34（运维）**：pip 强制重装后必查 `~xxx` 残留目录（transformers 半删除致 BACKENDS_MAPPING 报错）。
- **发现 35**：`cudaDeviceSynchronize` 逐调用同步是吞吐杀手——497 次 QLinear/step 全设备同步，改异步发射（消费端 `.item()` 统一同步）后上传 9.8×、prefill 2-2.6×、decode 1.24×。
- **发现 36**：decode 真瓶颈是 CPU dispatch（DeltaNet 纯 torch 回退，~2600 次 dtype 转换/step）而非 GEMV——GEMV 理论上限 26 tok/s，引出 R1/R2 两条路线。
- **发现 37**：Qwen3_5 部署三坑——rope 参数在 config.rope_parameters dict 需直接实例化拷贝 inv_freq；meta 骨架 to_empty 清空 non-persistent buffer；checkpoint 键 model.language_model.* vs 骨架 model.* 映射。
- **C1 补注（2026-08-17）**：embedding 层 NVFP4 CPU packed 落地——FP16 2.37GB → 0.68GB（packed 0.59GB + scales 0.09GB），**RAM 省 1.69GB**；`_qwen38_emb_pack.py` 打包 + `_qwen38_infer.py` 查表反量化，63/63 与 257/257 token 一致性双验收 PASS（推理无损红线达标）。

### 阶段 7：R1 vs R2 优化路线对决

| 方案 | tok/s | vs baseline | token 匹配 |
|---|---|---|---|
| baseline DynamicCache | 5.56-5.92 | 1.00× | — |
| **R2: fused DeltaNet kernel** | **10.27** | 1.78× | 63/63 |
| **R1: StaticCache + CUDA Graph** | **18.56** | 3.21× | 63/63 |
| **R1+R2 组合** | **21.78** | **3.77×** | **63/63** |

- **R2**（`nvfp4_dn_fused.cu` v2）：单 kernel 融合 decode 单步全部中间算子（conv1d update + silu + l2norm + recurrent + RMSNormGated），~100 次 aten op/层 → 2 次 launch/层。v1→v2 三个正确性修复：conv_state 是 4 帧滚动非 3 帧；拆 2 个 kernel（`__syncthreads` 无法跨 block）；fp16 舍入对齐（中间量 `.to(fp16)` 存取）。
- **R1**（`_qwen38_r1.py`）：transformers 5.14.1 StaticCache 原生支持混合架构且为 cudagraphs 设计（GPU tensor + 原地写）；warmup 污染 → capture 后快照恢复。
- **发现 38**：R1 潜力 > R2（纯 python 可移植 vs 逐位语义对齐）；组合后已达 GEMV 带宽瓶颈（46ms/step 中 GEMV 41ms）。
- **发现 39**：CUDA Graph 测试两陷阱——快照必须在 prefill 后立即抓（"未来状态"起跑输出连贯但全错）；graph 类 bug 用多步序列对照 + eager 同状态重放定位。

### 阶段 8：GEMV v3 LUT + MTP 投机解码对决

- **8A GEMV v3**（**发现 40**）：256 项 byte→half2 共享内存 LUT 消除反量化位操作；E8M0→float 纯位构造（float 指数 bias 与 E8M0 相同，bits=`sb<<23` 直通零成本）；`__launch_bounds__(256,8)` 锁 8 blocks/SM → **T=1 实测 419-518 GB/s（+37%）**；T=2 时 LUT 的 LDS bank conflict 反转（位构造版更快），fast_fwd 投影 GEMV2 化。
- **8B MTP**（EAGLE k=1 链式 T=2 投机）：257/257 正确性 PASS 但**净亏损**。修复三个正确性 bug：MTP 层 5 处实现错（对照 vLLM 逐段定位，教师强制 0%→52%）；**发现 41**：--check 基线自身 first 重复（"基线"错了而 MTP 是对的，三链对照定位）；**发现 42**：T2 拒绝回滚 off-by-one（tok_c 写 @wpos+1、cumlen 回退 1）。
- **发现 43（定论）**：MTP 盈利平衡点 p≥87% 不可达（单层 NVFP4 draft 教师强制 52%、实测接受 59%）——27B DeltaNet 混合架构 + 单层量化 draft 上，投机不敌确定性 R1+R2。

### 阶段 8C：GEMV 深度优化 D1（v4→v8 PRMT，发现 47-48）

- **探针宏定位法**（LUT_BYPASS/WBYPASS/ACT_BYPASS 编译期宏，关态零影响对照）：LUT 换常量后 470→837 GB/s。**发现 47**：LDS 随机 bank conflict 损失 43%，是 LUT 反查表方案的性能天花板；4 相 LUT 无效——随机索引下固定相位偏移不改变 bank 负载分布。
- **发现 48（PRMT 硬件查表，v8 定案）**：`__byte_perm` PTX 指令全程寄存器反量化，完全脱离 LDS——正幅值高字节表 8 码点（lo32={00,38,3C,3E}，hi32={40,42,44,46}）+ 符号 PRMT 交织（selector 0x5140/0x7362）+ 交织 word（selector 0x1001/0x3223/0x5445/0x7667 & 0xFF00FF00）+ 双 half2 累加链每 32 元素 float 结算 + E8M0 位构造（sb<<23）。调试关键：PRMT selector 仅用低 16 位，w 须拆低/高 16 位分别查表（`_debug_v8_prmt.py` 位级模拟定位）。
- **D1 验收（2026-08-17 落盘实测）**：v8 与 v4 三尺寸（256/2048/31232 × K=5120）**逐位一致**；vs FP64 真值 mean_rel 1.65e-03~2.52e-03、p99 ≤2.74e-02；带宽 M=5120/12288/24576/31232 → 618/716/778/**792 GB/s**——超 650 目标 22%，vs v3.2 LUT 518 GB/s **+53%**，达理论 900 的 88%。折算 decode：R1+R2 组合 46ms/step 中 GEMV 41ms → ~27ms，理论 decode 21.78 → ~26+ tok/s（待 E2E 复测确认）。

### 阶段 9：8K 上下文跑通（分块 prefill 笔记式架构）

- 2048+ prompt 必爆根因 = sdpa math backend 物化 L×L 注意力矩阵 + DeltaNet scan 中间量（非"全序列 logits"——反证实验推翻，归因误判已隔离为 nvfp4-024：相关性吻合≠因果证明）。
- **笔记式"随读随放"三层架构**：Layer0 草稿（GPU 末行 logits 0.3MB）→ Layer1 便签（CPU 全序列 hidden 84MB@8K）→ Layer2 按需查笔记（分块 lm_head，数学保证 bit-exact，实测 20/20）。
- **三连修复链**：分块 prefill 512/块（激活 ∝chunk 且更快 302→385 tok/s）→ vf_snap 48 层 DeltaNet 状态快照 → 14 层 KV 前缀快照（解锁 graph 捕获）。
- **实测**：plen=8090/maxlen=8192 全链 OK（prefill 424 tok/s、decode 15.4 tok/s、min_free=48MB）；安全边界 4K→**8K**。

### 阶段 10：视觉塔量化 P1（发现 44-46）

| 方案 | pooler cos / 行最低 | 判定 |
|---|---|---|
| NVFP4 全量（110 Linear） | 0.9525 / 0.7898 | FAIL |
| NVFP4 层混 / 行混 | 未过线 | FAIL |
| FP8 / INT8 全量 | 全局过线 | 行级真损伤 |
| **M1：L0-L4 FP16 + 深层 INT8 per-row** | **0.99988 / 0.9990，513MB** | **PASS 定案** |

- **发现 44**：同一 codec 文本塔 PASS（cos=0.993）视觉塔 FAIL——27 层堆叠误差累积 + 浅层敏感；量化验收必须整塔端到端。
- **发现 45**：全局 cos 掩盖行级真损伤——验收固化双指标（全局 ≥0.99 且行最低 ≥0.98）。
- **发现 46**：量化误差由权重分布决定与输入分布无关——合成噪声输入即可做验收。
- 工程三坑（已错误隔离 nvfp4-029）：`model.visual.*` 前缀剥离错致 strict=False 静默全 miss（随机权重跑通不崩溃）；嵌套生成器作用域；K=4304 非 64 倍数 reshape 崩溃。

---

## 三、配套工作：超长上下文推理架构方案评审

**吻合可信 4 条**：NVFP4 cos≈0.99；4→2bit 码点断崖；V100 软件模拟定位；NVFP4 商用/NVFP2 科研划分。

**四大硬伤**：Prefill 性能预估乐观 20-60 倍（40B 每 token ≈80 GFLOP，V100 MFU 30% ≈470 token/s，2048 token ≈4.4s 非 0.12-0.22s）；KV 复用 RoPE 位置冲突未答；"摒弃切片"与二级检索自相矛盾；1.58bit 40-44B 三值模型不存在。

**三处参数修正**：NVFP4 0.5→0.5625 B/param；E1M0 非零码点 3 个；两级 scale 结构（per-tensor + per-16 E8M0）。

### 对策定案（2026-08-17 回填，B + A1-A4）

**A1 Prefill 性能预估修正（实测锚定，禁理论峰值外推）**
- 原方案乐观 20-60 倍根因：按 TensorCore 峰值 FLOP/s 外推，忽略实测 MFU 与注意力 O(L²) 项。
- 实测锚点：本项目 27B NVFP4 实机 prefill **424 tok/s**（阶段 9 分块 512/块实测）；40B 每 token 计算量 ≈1.48×，线性外推 ≈286 tok/s → **2048 token prefill ≈7.2s**（原方案宣称 0.12-0.22s）。
- 对策：一切 prefill 指标以 27B 实测为锚外推并标注"V100 实测 MFU 口径"；方案文档禁用理论峰值外推；长上下文 prefill 走分块流水（阶段 9 方案）+ 三层架构摊薄重读成本（见 A3）。

**A2 RoPE 位置冲突 → 段级位置重映射（rebase）**
- 冲突：二级检索召回的历史 KV 段，缓存时位置编码 p_old 与当前序列位置不一致，直接拼接破坏 RoPE 相对位置语义。
- 对策：段级 rebase——召回段整体平移到当前序列基址 `pos_new[i] = base + i`（i 为段内偏移）。数学依据：RoPE 旋转矩阵正交，注意力分数 <q_{m+d}, k_n> 仅依赖相对位置；段内相对位置不变 → 段内注意力分数严格不变，跨段相对位置由 rebase 显式重定义。
- 实现约束：KV 缓存增存"段 id + 原位置 id"；旋转增量 Δθ=(base−p_old)·θ_i 每段离线预计算一次；与阶段 9 已验证的 14 层 KV 前缀快照机制直接复用。

**A3 架构自相矛盾 → 三层检索架构（指针制）**
- 矛盾：宣称"摒弃切片全量常驻"又依赖"二级检索"——若检索产物是明文切片，常驻承诺即破裂。
- 对策定案：**Layer0 粗导航**（摘要索引常驻 GPU）→ **Layer1 细检索**（只产出指针/段 id，CPU 侧）→ **Layer2 明文真相源**（命中时按需物化进上下文）。检索产物=指针，明文仅在命中时物化。
- 工程同构性：与阶段 9 已跑通的"草稿/便签/查笔记"三层完全同构（GPU 末行 logits 0.3MB + CPU 全序列 hidden 84MB@8K + 按需分块 lm_head），20/20 bit-exact 已验证——非纸面架构。

**A4 40B 三值量化无落地 → 16GB 上限 27B（数学证明）**
- V100 16GB − 激活/KV/CUDA 上下文 ≈14.4GB 权重预算；27B NVFP4 实测占用 14.42GB 恰好贴线（阶段 6）。
- NVFP4 0.5625 B/param → 16GB 工程上限 ≈25.6B（纯权重），叠加头层/敏感层 FP16 冗余 → **27B 为工程上限定案**。
- 40B 需 ≥22.5GB 权重 + 开销 → 32GB 硬件起步；1.58bit 三值（0.25 B/param 理论）在 40-44B 无公开成熟模型与工具链，属科研级风险，不入工程方案。
- 对策：本机模型规模钉死 27B；40B+ 需求走"云端大模型 + 本地 27B 摘要/缓存"混合，不在 V100 上追 40B。

**B 落地前置约束清单（三硬约束 + 参数修正定案，本项目实测坑固化）**
1. **scale 溢出约束**：E8M0 scale_byte 编码期必须 clamp ≤254（255 → `exp2f(128)` float32 溢出 inf → GEMM 全 NaN，发现 6/7）；两级 scale（per-tensor FP32 × per-16 E8M0）相乘后编码端验算 |dequant|_max ≤ 65504（half 上限），超限即拒绝该块并降级 per-tensor 缩放。
2. **累加器约束**：禁用裸 half 长链累加——GEMM 累加器必须 float32（发现 8-10：分段 FP16 累加持续舍入）；GEMV 用 half2 双累加链 + 每 32 元素 float 结算（v8 实测定案）；FP16 累加超 ~2¹¹ 元素精度必然崩塌。
3. **float64 真值约束**：内核正确性验收基准一律 FP64 点积——FP32 真值自身误差会掩盖 half 内核 1e-3 级误差结构；验收三件套 = mean_rel + p99_rel + max_abs（v8 范例：1.65e-03 / 2.74e-02 / 0.0018）。
4. 参数修正定案：NVFP4 显存 **0.5625 B/param**（4bit 元素 + 0.5bit E8M0 + per-tensor 冗余，实测 0.2812 比值精确贴合）；E1M0（NVFP2 科研线元素格式）非零幅值码点 3 个，与 NVFP4 的 E2M1（7 个非零码点 0.5-6.0）不可混用；两级 block scale 结构 = per-tensor FP32 + per-16 E8M0，非单级。

---

## 四、V100 硬件底层约束（最终版）

1. **无 cp.async**（Ampere 才引入）——双缓冲只能 `__syncthreads()` + 多 stage shared 手动管控；
2. **TensorCore 仅 FP16**——4bit 必须软件反量化为 FP16 fragment（位构造 ~3 条/元素是正确做法）；
3. **工具链自研**——无官方 NVFP4 算子，编解码/GEMM/GEMV/融合内核全自研；
4. **~~性能弱于 FP16~~ 已修正**：朴素内核阶段（v2）确为 FP16 的 55-70%；**v3 寄存器分块起 NVFP4 全面反超 FP16（v5 达 5.8-13.5×）**——显存带宽节省在访存受限负载下成为净收益。软件模拟的真正天花板是 kernel 效率（v5 30.5% TC 利用率），不是反量化开销。

---

## 五、知识库入库状态（三批累计）

| 批次 | 内容 | 库容变化 |
|---|---|---|
| 阶段 8（2026-08-17） | 20 条（架构3+算法6+代码模式4+基准2+错误隔离4+流程1） | 11,332 → 11,352 |
| 阶段 9 | 3 条（021r 归因修正 / 023 三连修复链 / 024 归因误判隔离）+ 补齐 2212 条历史 FTS 缺口 + 清理 1703 条 FTS 孤儿 | 11,352 → 11,357（FTS 覆盖率 100%） |
| 阶段 10（P1） | 5 条（定案/方法论/基准/错误隔离×2） | 11,357 → 11,362 |
| 阶段 11（D1/C1/D3，2026-08-17） | 6 条（基准 030/033 + 代码模式 031/032 + 架构 034/035） | 11,362 → **11,368** |

- nvfp4 系列 36 条（nvfp4-001..035 + 021r），向量 all-MiniLM-L6-v2 384 维 **100% 覆盖**，memories_fts 双写（阶段 11 顺带清理 cuda-007 重复行，mem=fts=11,368 对齐），F 盘镜像同步；
- 阶段 11 双路召回验证：FTS5 关键词 PRMT→031/030、探针宏→032、rebase→034、nvfp4-03x 精确命中全 PASS；
- **错误隔离 7 条**（⛔前缀 + 错误原文 + 机理 + 正确做法 + 定位方法）：MTP 层 5 处实现错 / T2 回滚 off-by-one / 基线 first 重复 / Graph 快照时序+闭包 / 归因误判 / NVFP4 视觉塔全量失效 / 工程三坑；
- 双路召回验证：FTS5 中文关键词 100% 命中；向量英文语义精准（cos 0.5+）；纯中文长句对 MiniLM 区分度弱（走 FTS5 通道）；
- 入库流程固化（nvfp4-020）：迭代 → 查重 → 清洗 → 错误隔离标记 → 向量+FTS5 入库 → F 盘镜像；**推理无损红线**：优化必须 token 一致，达不到回退上一最佳。

---

## 六、项目整体资产归档

### 编解码与内核
| 文件 | 用途 |
|---|---|
| `nvfp4_codec.py` | 编解码 Python 基准（sign 内嵌 v2，0.5625 B/param） |
| `nvfp4_cuda.cu` / `nvfp4_cuda_v2.cu` | 单缓冲 / 双缓冲融合反量化-GEMM |
| `nvfp4_baseline_f16.cu/.dll` | FP16 基线内核（隔离反量化净开销） |
| `nvfp4_cuda_v3/v4/v5.cu/.dll` | 寄存器分块 / __hfma2 / TensorCore wmma 五代优化 |
| `nvfp4_gemv.cu/.dll` | GEMV decode 内核（v8 PRMT 硬件查表，**792 GB/s**，D1 验收） |
| `_qwen38_emb_pack.py` | embedding NVFP4 CPU packed（2.37→0.68GB，RAM 省 1.69GB） |
| `_bench_stage3.py` / `fp4研究\nvfp4_cuda_v2s2/s3.dll` | D3 3-stage 对照（+14.1%，bit-exact） |
| `_verify_gemv_v8.py` / `_debug_v8_prmt.py` / `_bench_gemv_dll.py` | v8 正确性/位级调试/带宽基准工具链 |
| `nvfp4_dn_fused.cu/.dll` | R2 fused DeltaNet decode 内核（v2） |
| `fp4研究\nvfp4_*` | 阶段 2/3 归档副本 + CUDA 12.8.1 安装包 |

### 部署与推理
| 文件 | 用途 |
|---|---|
| `_qwen38_nvfp4_pack.py` | mmap 流式量化打包（25.63B → 13.42GB） |
| `_qwen38_infer.py` | E2E 推理（R1+R2 默认路径：build → prefill → fused patch → graph 捕获 → decode） |
| `_qwen38_r1.py` / `_qwen38_r2.py` | R1 / R2 实测脚本 |
| `_qwen38_mtp.py` + `_dbg_mtp2~5.py` | MTP 投机解码 + 诊断链 |
| `_qwen38_vramtest.py` | 长程显存实测 |
| `_smoke_gemv2.py` | GEMV2（T=2）基准 |

### P1 视觉塔
| 文件 | 用途 |
|---|---|
| `_p1_vision_nvfp4.py` | NVFP4 全量验证（FAIL 定案证据） |
| `_p1_diag.py` | 误差定位四步（分组/逐层/outlier/矩阵） |
| `_p1_hybrid/_rowmix/_input_ab/_fp8/_rowerr.py` | 层混/行混/输入对照/FP8-INT8/行误差实验链 |
| `_p1_final.py` | M1 定案脚本（cos=0.99988） |

### 知识库工具链
| 文件 | 用途 |
|---|---|
| `_import_nvfp4_to_cezanne.py` | 标准入库管线（查重→向量化→双写→镜像） |
| `_kb_dedup_check.py` / `_verify_nvfp4_kb.py` | 查重 / 召回验证 |
| `_p1_kb_entries.json` + `_p1_verify_kb*.py` | 阶段 10 条目源 + 验证 |
| `NVFP4_研究记忆.md` | 全过程成果与发现总记录（阶段 1-10） |

---

## 七、整体汇总核心结论（最终版）

1. **格式与内核完全可靠**：NVFP4 数学逻辑自洽，自研编解码/CUDA 内核经小矩阵→真实尺寸→整塔→实机四级验证；显存压缩 0.2812 精确贴合理论，27B 模型 14.42GB 塞进 16GB V100。
2. **"软件模拟慢于 FP16"已被五代 kernel 优化推翻**：v3 寄存器分块是分水岭（计算/访存比 ×4 后反量化被掩盖），v5 TensorCore 达 12.8× vs FP16；反量化净开销从理论 30-45% 修正为实测 +4.7%（位构造版）。
3. **与 Blackwell 差距 236× 但数值语义 1× 等价**：编码器/验收标准/方法论 100% 可迁移；软件损耗 3.28× 是 V100 侧优化空间。
4. **推理系统结论**：decode 瓶颈排序 = CPU dispatch（R1 CUDA Graph 根治，3.21×）> kernel 内真实执行（R2 融合根治，+17%）> GEMV 带宽（**792**/900 GB/s，v8 PRMT 定案，88% 理论）；MTP 投机在本配置净亏损（p≥87% 不可达）。
5. **多模态量化策略不可复用**：文本塔 NVFP4 可用、视觉塔失效；分层混合精度（浅 FP16 + 深 INT8）是最优路径；验收必须整塔端到端 + 行最低 cos 双指标。
6. **可复用工程资产**：全套编解码/GPU 内核/无硬件前置验证模拟器/Windows CUDA 编译避坑规范（BOM、dllexport、管道截断）/量化知识库标准化入库流程（36 条，错误隔离 7 条）。

---

## 八、遗留待办（真实 open items）

| 优先级 | 待办 | 依据 |
|---|---|---|
| ~~P0~~ ✅ | ~~P2 多模态实装~~ **已完成（阶段 11）：vision_m1.npz 打包 + VLinear 前向反量化 + mrope 4D 位置，视觉 cos=0.99988/行最低 0.9991，文本路径 96 token 全一致** | 阶段 10 → 阶段 11 实装 |
| ~~P1~~ ✅ | ~~GEMV 带宽 518 → 逼近 900 GB/s 理论~~ **D1 已验收：792 GB/s（88% 理论）**；剩余 12% 需 ncu 深剖，性价比自评 | 阶段 8C 发现 47/48 |
| ~~P1~~ ✅ | ~~embedding NVFP4 化（省 2.37GB RAM）~~ **C1 已完成：RAM 省 1.69GB，token 一致性双验收 PASS** | 阶段 6 C1 补注 |
| ~~P1~~ ✅ | ~~E2E decode 复测（理论 ~26+）~~ **阶段 11 实测：R1+R2 = 32.02 tok/s（v1 25.92，+23.5%），超额达标；v1/v8 生成逐字一致 63/63** | 阶段 11 |
| ~~P2~~ ✅ | ~~MTP draft 保精度（FP16/Q8）~~ **阶段 11 定论关闭：FP16 draft 542MB OOM；Q8 draft（275MB 分块反量化）接受率 62%（NVFP4 58%），瓶颈=MTP 头预测上限非量化；端到端 22.5 < 32.02 倒挂** | 阶段 11 |
| ~~P2~~ ✅ | ~~3-stage TC 内核复验；mma PTX 内联~~ **阶段 11 双定论：3-stage 不采纳（平均 -1.4%，bit-exact 过，TC 计算密集 2-stage 已覆盖 LDG 延迟）；mma PTX 物理不可行（ptxas 实证 m16n8k8 需 sm_75+、m16n8k16 需 sm_80+，Volta 只有 wmma）** | 阶段 11 |
| P3 | Blackwell 侧 CUTLASS SM100 block-scaled GEMM 适配 | 发现 31（kernel 不移植，配置化 ~500→~100 行 [估]） |

> 注：原文档"阶段 4/5 待执行"为状态过时记录，实际均已完成（见 1.3 表）。

---

## 阶段 12：前缀 KV+DeltaNet 状态跨请求复用 P4a（进行中，2026-08-26）

> 触发：对照《拆一座AI数据中心》四问（哪些计算不用重做/哪些等待可消除/哪些算力没吃满/哪些资源没协同）审视现有栈。连续批处理/Prefill-Decode 分离/投机采样三项已有定论或不适用，唯一未开发空间 = 前缀 KV 复用（"推理税"）。

### 12.1 设计（已定稿，等 GPU 恢复后实测）

- **机制**：固定前缀（system prompt/文档）prefill 一次 → 快照（14 层 attention KV + 48 层 DeltaNet conv/recurrent + cumulative_length）驻 CPU → 后续请求 restore 后仅 prefill 后缀。单机版 RadixAttention 第一步（无基数树，单前缀字典）。
- **地基复用**：`refresh_deltanet_state`（每请求 prefill 后刷 _vf 缓冲）+ `GraphDecoder.set_pos`（全局 graph 复用）+ `snap/restore_decode_state`（KV 快照）——服务器多请求机制早已齐备，本阶段只补"前缀快照→restore→跳过前缀 prefill"一环。
- **bit-exact 关键约束**：前缀长度必须是 CHUNK(512) 倍数——保证 DeltaNet 分块 scan 边界与全量 prefill 完全一致，否则浮点舍入路径不同。
- **验收设计**（`fp4研究\_qwen38_p4a_prefix.py`，4-run 对照）：
  - Run1/Run2 baseline A/B：全量 prefill（前缀+后缀A/B）+ decode 63 tok；
  - Run3 冷请求：prefill 前缀→快照→prefill 后缀A，生成须与 Run1 逐位一致；
  - Run4 热请求：restore 前缀快照→仅 prefill 后缀B，生成须与 Run2 逐位一致；
  - **打毒测试**：run 间 KV 填 12345.0（有限大值，规避 NaN×0 掩码歧义）+ DeltaNet 状态清零，restore 任何遗漏→token 必然发散；
  - 指标：首 token 成本 冷 vs 热（预期 1024 tok 前缀省 ~2.4s/请求）+ 快照一次性成本。
- **预期效益**：固定前缀多轮场景首 token 延迟从 ~2.5s（424 tok/s × 1057 tok）降至 ~0.2s（restore + 33 tok 后缀）。

### 12.2 阻塞（2026-08-26/27 GPU 故障，✅ 已解除）

- **V100 驱动故障**：错误代码 10（CM_PROB_FAILED_START，**0xC000009A = STATUS_INSUFFICIENT_RESOURCES**）。
- **排查链**：① 提权 restart-device ✗；② 官方驱动全新重装 ✗（573.76，MD5 校验通过，静默安装 272s）；③ 系统重启 ✗；④ disable/enable ✗；⑤ 诊断定论 **PCI BAR 未分配**（`Win32_PnPAllocatedResource` 中 V100 零资源）。
- **根因与解决**：BIOS **Above 4G Decoding 被关**（V100 16GB BAR 必须 64-bit 映射）→ 用户开启后，`pnputil /remove-device` + `/scan-devices` 强制重新协商 BAR → PnP OK → 重启后 NVML/CUDA 完全恢复（TCC，573.76，16.7GB free）。
- **教训**：V100 装机必查 Above 4G Decoding；主板掉电/BIOS 重置会静默关闭该选项，症状是驱动重装也救不回的错误代码 10。

#### 工程沉淀：NVIDIA 驱动 CDN 直链下载方法（可复用）
- 直链格式：`https://us.download.nvidia.com/Windows/Quadro_Certified/{ver}/{ver}-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe`
- **裸 curl 必 403**（Akamai 301→.cn 后 Zen 服务器拒绝）：必须带浏览器 UA + `Referer: https://www.nvidia.com/en-us/drivers/details/{id}/` 即可 206/200
- 573.76 文件名/MD5/大小：`573.76-quadro-rtx-desktop-notebook-win10-win11-64bit-international-dch-whql.exe` / 73A8091337B0C9EF4DB545BE3A3DD0EC / 682,707,496B；V100 用的 INF 为 nv_dispwi.inf（oem4.inf）

### 12.3 P4a 实测结果（2026-08-27，✅ 全部 PASS）

**验收（推理无损红线：token 逐位一致）**：

| 对照 | 结果 |
|---|---|
| Run3（冷+快照）vs Run1（baseline A 全量 prefill） | **PASS** 64 token 逐位一致 |
| Run4（热=restore 复用）vs Run2（baseline B） | **PASS** 逐位一致 |
| 打毒测试（run 间 KV 填 12345.0 + DeltaNet 状态清零） | **PASS** restore 零遗漏 |

**性能（512 tok 固定前缀 + 30 tok 后缀）**：

| 指标 | 冷（全量 prefill） | 热（restore + 仅后缀） |
|---|---|---|
| 首 token 成本 | 1.81s | **0.71s（省 61%）** |
| restore 成本 | — | 33ms |
| 快照一次性成本 | 97ms | — |

**发现 50：前缀复用状态集 = attention KV + DeltaNet conv/recurrent，无 cumulative_length**
- 混合架构 StaticCache：`cumulative_length` 是 attention 层属性，DeltaNet 层无此属性（首次 snap 实现踩坑）；`conv_states`/`recurrent_states` 在部分层为 dict 而非 tensor（打毒需 isinstance 守卫）。
- 前缀必须 CHUNK(512) 对齐的约束**验证有效**：512 对齐下 DeltaNet 分块 scan 边界一致，浮点舍入路径完全一致 → token 逐位一致。

**发现 51：前缀复用收益随前缀长度线性增长**
- 512 tok 前缀省 61%（1.81→0.71s）；restore 成本 33ms 与前缀长度无关（纯 memcpy）。
- 外推 2K 前缀（阶段 9 场景）：冷 ~4.4s → 热 ~0.8s，省 ~82%；8K 前缀省 ~90%+。
- 与文章《拆一座AI数据中心》"推理税"论断定量吻合：固定前缀场景的重复 prefill 是最大可消除浪费。

**下一步（P4a+）**：多前缀字典（单前缀 → N 前缀 LRU）；快照驻留策略（CPU RAM 每 1K 前缀 ~60MB，8K 前缀 ~480MB 需权衡）；与 A2 段级 rebase 打通（跨请求 KV 段复用）。

### 12.4 P4b step 级 profile 结果（2026-08-27，`_qwen38_p4b_profile.py`）

**方法**：手动复刻 `GraphDecoder.step()` 逐段 CPU 计时 + CUDA event 测 GPU 侧 replay 真实时长；512 步持续负载（排除首轮页错误：CPU gather 3.43→0.25ms）。

**分解（每 step 40.3-40.9ms，24.4-24.8 tok/s，3 次稳定复现）**：

| 环节 | 耗时 | 占比 |
|---|---|---|
| **GPU 侧 graph replay（event 实测）** | **39.84ms** | **~97%** |
| CPU gather（NVFP4 embed 行反量化，CPU） | 0.25-0.86ms | <2% |
| replay CPU 启动 | 1.19-1.37ms | ~3%（与 GPU 重叠前） |
| H2D（pinned→static_embeds） | 0.06ms | ~0% |
| pos4.fill | 0.03ms | ~0% |

**发现 52：decode 已 97% GPU-bound，CPU 侧无剩余优化空间**
- R1 CUDA Graph 已根治 CPU dispatch（阶段 7 定论在此定量确认）；graph 外 CPU 合计仅 ~1.6-2.2ms/step，且被 `.item()` 同步串行化掩盖。
- CPU gather 是 graph 外最大项（NVFP4 packed embedding 行查表反量化在 CPU 做），搬 GPU/缓存的上限收益 ~2%，性价比低，不做。

**发现 53：真正残余在 graph 内部——GEMV 之外 GPU 工作约 21.6ms**
- GEMV 理论：14.42GB ÷ 792GB/s ≈ 18.2ms；replay 实测 39.84ms → **非 GEMV GPU 工作约 21.6ms（54%）**：DeltaNet torch 回退算子（发现 36 的 ~2600 次 dtype 转换未被 R2 融合内核完全消除的部分）+ attention KV + elementwise。
- 这 21.6ms 是下一个优化主战场，需 ncu graph 内剖分定位（对应遗留待办"GEMV 剩余 12% 带宽深剖"的扩大版：不是 12%，是 54% 的非 GEMV 在途时间）。

**发现 54（勘误）**：本项目曾引用"R1+R2+v8 = 32.02 tok/s"为误记（主文档与记忆库均无此数）。实际记录：R1+R2+GEMV v3.2 = 25.89 tok/s（当前最佳基线）；v8 E2E 理论 ~26+ tok/s。本次 24.79 tok/s（MAXLEN=3072 短上下文、驱动重装后）与基线差 ~4%，属条件差异，非回归。**教训：性能结论必须以本档记录数字为准，跨会话引用前先查库。**

### 12.5 NCU 剖析 decode graph 内核（2026-08-27，`_qwen38_p4c_ncu.py` + `_ncu_parse.py`）

**工具链沉淀（发现 55）**：
- 本机无 NCU → Nsight Compute 2025.2.1 MSI 直链下载（`developer.nvidia.com/downloads/assets/tools/secure/nsight-compute/...msi`，需 UA+Referer）静默安装 `msiexec /i /qn /norestart`。
- **Windows 上 NCU 剖析 GPU 性能计数器必须提权**（ERR_NVGPUCTRPERM）；提权 Start-Process 传中文路径会失败 → 脚本/cmd 全部放纯英文路径（`C:\Users\42235\`）。
- **CUDA Graph 剖析**：`ncu --graph-profiling=node --metrics gpu__time_duration.sum --csv --page raw`；CSV 中按"倒数第 3 步首个 v8 GEMV"切分 replay 段（44954 条内核中切出 5831 条 = 3 步）。
- NCU 计时（30.30ms/步）与 wall-clock（39.84ms/步）有差（时钟/序列化），**占比归因有效，绝对值以 wall-clock 为准**。

**每步内核归因（发现 56，NCU 口径）**：

> ⚠️ **勘误（阶段 14 发现 65）**：本表绝对值被旧解析的步切分 bug 低估 1.5 倍（尾段实为 2 整步+截断，÷3 应为 ÷2）。修正：GEMV 实为 ~21.6ms/步（**663GB/s，非"已达 792 接近极限"**）、elementwise ~11.6ms、attention ~7.5ms。占比列仍有效。

| 类别 | μs/步(旧÷3) | μs/步(修正÷2) | 占比 | 判定 |
|---|---|---|---|---|
| GEMV v8 | 14402 | **21603** | 47.5% | 修正后 663GB/s（隔离基线 792），仍有 ~20% in-graph 空间 |
| **torch elementwise** | **7721** | **11582** | **25.5%** | ⭐ 发现36 dtype 转换残余（~1583 次/步小内核），R2 只融合了 dn_rec/dn_conv |
| **attention fmha_cutlassF 32x128** | **5010** | **7515** | **16.5%** | ⭐ T=1 小序列（~21 tok KV）MemEff kernel 严重低效，理论应 <100μs |
| dn_rec 融合 | 2333 | 3500 | 7.7% | R2 已优化 |
| torch reduce + 其他 | 829 | 1244 | 2.7% | 小头 |

**优化路线判定（P5 候选）**：
- **B（attention）**：换 flash/手写 GQA T=1 kernel，预期消 ~4.9ms/步；
- **A（elementwise）**：dn_rec 扩展融合（把 Norm/rsqrt/pow/add/silu 链并入），预期消 ~6ms/步；
- 合计每步 30.3→~19ms（NCU 口径）/ wall-clock 39.84→~29ms → **~34-35 tok/s**，与 35+ 目标吻合。
- GEMV 继续压榨的边际收益 <1ms（88%→95% 理论），性价比低于 A/B。

### 12.6 P4a+ 多前缀 LRU 缓存（2026-08-27，`fp4研究\_qwen38_p4a_lru.py`，✅ 全部 PASS）

**设计**：单前缀 → `OrderedDict` 多前缀 LRU（单机版 RadixAttention 第二步）。
- key = 前缀 token 序列 SHA256 前 16 位（token 级，非文本级）；value = (KV 快照, DeltaNet 快照) CPU RAM 驻留；
- 驱逐 = 字节容量上限（343MB ≈ 2 前缀快照）超限逐出最久未用（`popitem(last=False)`）；
- 验收：3 前缀 P1/P2/P3 × 服务序列 `[P1, P2, P3, P2, P3, P1]`（MISS×4 / HIT×2 / 驱逐×2 / 被驱逐后重建）× 63 token 与全量 prefill baseline 逐位对照 + 打毒。

**结果**：

| 指标 | 值 |
|---|---|
| 逐位一致 | **6/6 全部 PASS** |
| LRU 统计 | hit=2 miss=4 evict=2 驻留=2 |
| HIT restore | **70ms**（vs MISS 前缀 prefill+快照 ~1.5s，**省 ~95%**） |
| 快照成本 | 172MB/前缀（512 tok） |

**发现 57：`StaticLayer.update()` 忽略传入的 `cache_position`，用自身 `cumulative_length` 定位 KV 写入槽位**
- transformers 新版 cache API：`cache_position = torch.arange(kv_len) + self.cumulative_length` 后 `index_copy_`（[cache_utils.py L420-427](C:\Users\42235\AppData\Local\Programs\Python\Python312\Lib\site-packages\transformers\cache_utils.py#L420-L427)）——模型 forward 传的 `cache_position` 参数被 `*args/**kwargs` 吞掉。
- **graph decode 内 `cumulative_length.add_(1)` 被 CUDA Graph capture，每次 replay 自动 +1**；`GraphDecoder.set_pos` 只拨 `static_pos`，不重置 cache 层的 `cumulative_length`。
- 后果链（P1 FAIL 根因）：LRU 版重写 `poison_cache` 时丢失了原 P4a 的 `cumulative_length.zero_()` 归零 → 下一次 prefill 的 KV 从上次 decode 残留位置（L+N）写入 → 槽位错乱 → attention 读到打毒值 12345 → logits 退化首 token=0 → 累计越过 max_cache_len(3072) 触发 `index_copy_(): index out of bounds` 断言。
- **教训：跨请求复用 StaticCache 时，"重置状态集"必须包含 `cumulative_length`；打毒函数是防回归锚点，重写时逐行 diff。**

**发现 58：服务序列设计必须与容量联算，否则验收指标（hit≥1）不可达**
- 错误序列 `[P1,P2,P3,P1,P2]`（容量=2）：P1 重建必驱逐 P2 → P2 永远 MISS（hit=0）。
- 正确序列 `[P1,P2,P3,P2,P3,P1]`：P3 入驻驱逐 P1 → P2/P3 命中刷新 LRU 位 → P1 重建驱逐 P2 → hit=2 miss=4 evict=2，三种路径（命中/驱逐/重建）全覆盖。

---

## 阶段 13：P5-B attention T=1 kernel 替换（2026-08-27，✅ PASS）

### 13.1 设计（`fp4研究\nvfp4_attn_decode.cu`，sm_70）

- **目标**：消除 12.5 NCU 归因中 attention fmha_cutlassF 的 5010μs/步（16.5%）——T=1 小 KV 序列下 MemEff kernel 严重低效。
- **kernel**：GQA T=1 两遍法 online-softmax。grid=24 blocks（每 q-head 一个）× 256 threads；
  - Pass1 每 thread 串行点积一行 k（512B 连续读，L/256 轮）→ scores 存 shared（12KB fp32）；
  - block reduce m=max / l=Σexp（两级：warp shfl → `s_red[warp]` → warp0 跨 warp 归约）；
  - Pass2 每 thread 负责一个 d 维，遍历 i 累加 `p_i·v_i[d]`（256 threads 同 iter 读同一 i 的 512B 行 → 完美 coalesced）。
- **graph 兼容**：L 从 `cumulative_length` device 指针动态读（graph replay 时随 add_(1) 自动变化，无需重捕获）；mask 语义 = attend `[0, L)`。
- **布局**：keys/values = StaticCache `(1, 4, 3072, 256)` fp16 连续；q 经 transpose 后内存仍每头连续 256；GQA 映射 `kv = h/6`（HQ=24, HKV=4）。
- **接入**：`_qwen38_infer.py::patch_attn_decode`（T=1 走自研 kernel，T>1 prefill/MTP 走原 sdpa）；q_proj 输出拆 query/gate（Qwen3.5 gated attention），kernel 输出 × sigmoid(gate) 后 o_proj。
- **shared 尺寸绑定 MAXLEN=3072**：launcher 对 max_len≠3072 返回 -2 拒绝。

### 13.2 结果（`fp4研究\_qwen38_p5b_attn.py`）

| 指标 | baseline (sdpa) | P5-B kernel |
|---|---|---|
| token 逐位一致（63 tok） | — | **PASS**（与 baseline 完全一致） |
| decode 速度 | 24.8 tok/s (40.3ms/步) | **28.8 tok/s (34.8ms/步)** |
| 每步节省 | — | ~5.5ms（≈NCU 预估 4.9ms，吻合） |

- 距 35 tok/s 目标还差 P5-A elementwise 融合（~6ms/步）。
- 精度：score/softmax/累加全 fp32，与 MemEfficient 累加顺序不同但 63 token 逐位一致（greedy argmax 鲁棒）。

### 13.3 发现 59：`__shfl_down_sync` 只在 warp 内归约（跨 warp 必须两级）

- v1 kernel 的 block reduce 直接 `for (o=128; o>0; o>>=1) shfl_down(...)` 后 `if (t==0) s_red[0]=m`——**shfl 的 width 默认 32，delta>31 的 shuffle 是 no-op**（返回自身值），实际只有 warp0 的部分和进入 `s_red[0]`，其余 7 个 warp 丢失。
- 症状：softmax 分母错 8 倍量级 → 注意力输出错 → **decode 全部输出 token=0**（首 token 来自 prefill 原路径所以正确，极具迷惑性）。
- 正确写法：warp 内 shfl（o=16..1）→ `if (lane==0) s_red[warp]=v` → `__syncthreads` → warp0 读 `s_red[0..7]` 跨 warp 归约（o=4..1）→ broadcast。

### 13.4 发现 60：nvcc `-shared` Windows 不自动导出 C 符号

- DLL 编译成功但 ctypes 报 `function 'launch_attn_decode' not found`；dumpbin /exports 只有 `NvOptimusEnablementCuda`。
- 根因：Windows nvcc -shared 生成的 .def 只含 NV 内部符号，**host 入口必须显式 `__declspec(dllexport)`**（本项目所有历史 DLL 均如此，新写 .cu 时易漏）。

### 13.5 发现 61：CUDA 静默装组件会掏空已有 toolkit 的 include

- 之前 `-n -s nvcc_12.8` 只装了编译器组件，v12.8\include 只剩 `crt/fatbinary_section.h/nvPTXCompiler.h`，**无 cuda_runtime.h/cuda_fp16.h**（编译报 C1083，与"pip cuda_runtime headers 缺 nv/target"连锁）。
- 修复：离线包补装 cudart 组件 `cuda_12.8.1_572.61_windows.exe -n -s cudart_12.8`（headers 恢复）。
- 附带：`patch_fused_deltanet` 必须在 prefill 之后调用（从 cache 抓 conv/recurrent 初始状态，此前 conv_states 为 None 直接 AttributeError）。

### 13.6 P5-A：elementwise 融合（2026-08-27，✅ PASS 32.2 tok/s）

**P5-A-3 定位（`_ncu_parse_ew.py` + `_ncu_parse_seq.py`，按 kernel 序列位置归属）**：
| elementwise 项 | 每步 | 归属 |
|---|---|---|
| RMSNorm 链 ×161 处 | ~5.5ms | 64层×2 + final + 16×q/k_norm；每处 10 内核，含**每步重算 `w.float()` 和 `1.0+w`** |
| RoPE 应用（neg/cat/mul 链）×16 | ~0.8ms | `apply_rotary_pos_emb` 内 rotate_half |
| sigmoid+mul ×16 | ~0.1ms | P5-B 残留 torch op |

**P5-A-1（attn kernel v2）**：`sigmoid(gate)·out` 并入 `attn_decode_kernel` 尾部，位精确复刻 torch 舍入链：`f16(f32(round_f16(acc)) · f32(round_f16(σ)))`（sigmoid/mul 内部均 fp32 opmath + 两次 f16 舍入）。

**P5-A-2（`nvfp4_rmsnorm.cu` 全站融合）**：
- 语义（modeling L749-754，zero-centered）：`out = f16(f32(x)·rsqrt(mean(x²)+eps)·(1+w))`；`w1 = 1+w` patch 时预计算一次（消除每步 weight cast/add）。
- grid=rows（每 block 一行），block = n≥1024 ? 1024 : 256；两级 warp 归约 Σx²；x 允许 chunk view 行间 gap（`stride(-2)` 传参）。
- **位精确**：单元测试 5120 维与 256 维 chunk 两用例 maxdiff=0、非零=0；63 token 逐位一致 PASS。
- 接入 `_qwen38_infer.py::patch_rmsnorm`（161 处 module 实例级 patch）。

**结果**：
| 指标 | baseline | P5-B | P5-A |
|---|---|---|---|
| token 逐位 | — | PASS | **PASS** |
| 速度 | 24.8 | 28.8 | **32.2 tok/s**（31.09 ms/步） |

### 13.7 发现 62：graph 内 kernel patch 必须 T>1 回退，否则缓冲驻留 OOM

- `patch_rmsnorm` 初版对任意 shape 都走 kernel → prefill 时每个 norm 模块保留 `(T=512, 5120)` 输出缓冲，161 处 × 5MB ≈ 650MB 驻留 → 第二次 prefill OOM（V100 16GB 本就剩 1.51GB）。
- 规则：**decode-only patch 统一 `if x.shape[1] != 1: return orig(x)`**（norm/attn 同策略）。graph 只 capture T=1 路径，prefill 走原路径零风险。

### 13.8 发现 63：MSVC 代码页 936 会误读 UTF-8 中文注释（C4819 → 符号丢失）

- `nvfp4_rmsnorm.cu` 首版 UTF-8 无 BOM：GBK 解码把注释字节与换行粘连，`#include <cuda_fp16.h>` 被吞 → `__half is undefined`（attn_decode.cu 恰好未触发，纯属运气）。
- 修复：**统一存 `utf-8-sig`（带 BOM）**，MSVC/nvcc 识别 BOM 后按 UTF-8 解析，中文注释无损。
- 附带：v2 签名改 6 指针后 `argtypes` 必须同步 `* 6`，否则 ctypes 把 `c_int` 位当指针报 TypeError（argument 6）。

---

## 阶段 14：P6 attention kernel v3 + NCU 步切分勘误（2026-08-27，✅ PASS 33.9 tok/s）

### 14.1 P6 设计与回退（v3 → v3.1，`nvfp4_attn_decode.cu`）

**v3 三项优化**（软件预验证 Rubin "exp 硬件吞吐加强"论述，实际被精度否决一项）：
1. `expf → exp2f`：SM70 上 expf 走软件模拟（~20 指令），exp2f 直达 MUFU.EX2 硬件单元；`scale·log2e` 预乘进 score；
2. **p 预计算**：v2 Pass2 每 thread 重算 `exp(s-m)·inv_l`（L×256 次 exp）→ reduce 后并行算一遍 p[i] 原地写回 s_sc（L 次），Pass2 变纯乘加；
3. **Pass1 half2 加载**：k 行点积 load 指令减半（256→128），half2→float2 转换位精确。

**v3 验收 FAIL → 根因 → v3.1**：
- v3（exp2f 版）63 token 验收 **FAIL**：首 8 token 与 baseline 一致，后续发散；
- **发现 64：exp2f((s·scale·log2e)−m) 与 expf(s·scale−m) 的 ulp 级差异经 63 步贪心解码链式放大**——softmax 分子分母同源抵消的假设对 argmax 临界 token 不成立（"greedy 鲁棒"假设被证伪）。半保留项 half2 加载与 p 预计算每 thread 计算同一表达式、值位相同，均位安全，FAIL 归因隔离到 exp2f 单项；
- v3.1 回退 expf（保留 2、3 两项）：**63 token 逐位 PASS，33.9 tok/s（29.47ms/步）**。expf 软件模拟成本已被 p 预计算压到 L 次/头（614 tok 上下文 ~24 头 × ~600 次），attn kernel 整体仅 0.19ms/步（0.7%），exp2f 理论收益 <50μs/步——**精度优先于速度，正确取舍**。
- 教训：**舍入链改变的"ulp 级无损"论证只对单次前向成立；多步自回归解码会放大任意 ulp 差异，位精确验收必须按完整 63 token 链执行。**

### 14.2 发现 65：NCU 解析步切分 bug（1.5 倍系统性低估）

**症状**：v3 NCU 解析报"每步 19.12ms"，与 wall-clock 29.47ms 矛盾（gap 10ms，graph replay 不应有 34% 空洞）。

**根因**：`step_first_v8[-3]` 取"最后 3 个步首"切 3 步段，但 **CSV 在最后一步中途截断**（NCU 停采）→ 最后一个"步首"是截断步起点，段实为 2 整步 + 3 个游离内核，÷3 低估 1.5 倍。

**三重裁决证据**（v3 NCU CSV 已删，全部旁证）：
1. 全文件 v8 总数 2983 = **6 步 × 497 + 1**（GraphDecoder `n_warmup=3` + 3 replay，代码侧核实执行步数恰为 6）；
2. 尾段 [33153, 35830) 含 996 个 v8 = **2.006 步**（996 ÷ 2 = 498/步）；
3. eager 剖析（`_qwen38_p6_prof.py`，无 graph capture）的 `aten::empty` 533 次/步 ≈ **497 个 QLinear 输出** + 36 其他，`aten::reshape` 1012 ≈ 2×497+18（QLinear 每次 forward 恰 2 个 reshape）——**每步 v8 GEMV = 497 实锤**。

**修正后每步分布（NCU 口径，v3 kernel，尾段 2 整步）**：

| 类别 | μs/步 | 占比 | 备注 |
|---|---|---|---|
| GEMV v8 | 21730 | 75.8% | 497 次 × 43.7μs；**14.42GB/21.73ms = 663GB/s**（峰值 900，隔离基线 792） |
| DN 融合 (dn_conv+dn_rec) | 3678 | 12.8% | 64 次/步 |
| torch elementwise | 2037 | 7.1% | RoPE 链/MLP silu·mul/index_copy 等 |
| rmsnorm (v_fused) | 1008 | 3.5% | 107 次/步 |
| attn (v3) | **194** | **0.7%** | 11 次/步，P5-B 4.9ms → 0.19ms（25 倍） |
| reduce + 其他 | ~20 | 0.1% | |
| **合计** | **28680** | 100% | **wall-clock 29.47ms → graph 利用率 97.3%** |

**阶段 12.5 勘误**：旧表 GEMV 14402μs/步（÷3 假值）隐含 995GB/s > 硬件峰值，不可能；修正 ÷2 后 21.6ms/步（663GB/s），与本次 21.73ms 跨运行一致（同一 kernel）。**"GEMV 已达 792GB/s 接近极限"结论作废**——in-graph 有效带宽 663GB/s，距隔离基线还有 ~20% 空间。

**方法论沉淀**：
- NCU CSV 步切分必须先核验全文件 v8 总数 = 步数 × 497 + 余数（余数>0 即末步截断，剔除）；修正版 `_ncu_parse_p6.py` 已实现。
- torch.profiler 对 graph replay 内核不可见（整图一个事件）；进程内做过 CUDA Graph capture 后 CUPTI 静默失效（0 事件）——**eager 剖析必须在无 capture 进程中做**，且 aten op 计数（empty/reshape）可反向核验 QLinear 调用数。
- NCU 提权（UAC）不可用时，eager aten-op 剖析 + 代码侧步数核算可作为分布与一致性的廉价裁决手段。

### 14.3 P5+P6 后的战场判定

- 优化史：baseline 24.8 → P5-B 28.8 → P5-A 32.2 → **P6 v3.1 33.9 tok/s**（单次运行方差 ±1.5，P5-A 复测亦见 34.1）；
- attention 已从 16.5% 降到 0.7%（fmha_cutlassF 4.9ms → 自研 kernel 0.19ms），**关闭**；
- **GEMV 75.8% 绝对主导**：in-graph 663GB/s vs 隔离 792GB/s——差距 ~5ms/步的可能来源：NCU kernel 隔离测量口径 / in-graph 调度干扰 / L2 互扰，需 NCU 提权重跑（`_run_ncu3.bat`，待 UAC）裁决；
- 次级目标：elementwise 2.0ms（RoPE neg/cat 链 ~0.9ms、MLP silu·mul ~0.6ms）+ rmsnorm 1.0ms + DN 3.7ms；
- 35 tok/s（28.6ms/步）缺口仅 0.9ms：GEMV 若在图内恢复 700GB/s 即达标；或 RoPE 融合进 attn kernel（~0.8ms）。

## 阶段 15：P7 真实 shape 隔离裁决 + 成对 GEMV 融合（2026-08-27，✅ PASS 35.3 tok/s）

### 15.1 P7-1 发现 66：663 vs 792 GB/s 之争裁决——尺寸效应为主、干扰为辅

旧隔离基线 792GB/s 只测过 4 个大 shape，而每步实际 launch 497 个 GEMV、9 种尺寸。对全部真实 shape × 真实次数做隔离 bench（`_bench_gemv_real_shapes.py`，小权重多 buffer 轮转防 L2 命中虚高）：

| shape | 次/步 | μs/核 | GB/s | ms/步 |
|---|---|---|---|---|
| 248320×5120 (lm_head) | 1 | 810.8 | **883** | 0.81 |
| 5120×17408 | 64 | 66.9 | 750 | 4.28 |
| 17408×5120 | 128 | 62.4 | 804 | 7.99 |
| 12288×5120 | 16 | 46.8 | 757 | 0.75 |
| 10240×5120 | 48 | 39.8 | 742 | 1.91 |
| 6144×5120 | 48 | 29.7 | 597 | 1.42 |
| 5120×6144 | 64 | 27.1 | 654 | 1.73 |
| 1024×5120 | 32 | 9.9 | **300** | 0.32 |
| 48×5120 | 96 | 8.1 | **18** | 0.78 |
| **隔离加权** | 497 | — | **722** | **19.99** |

**发现 66**：
1. **792 只是大 shape 瞬时值**——尺寸加权后的真实隔离极限是 19.99ms（722GB/s）。小 shape 严重欠占用：48×5120（48 warp / 80 SM）仅 18GB/s，8.1μs 里几乎全是延迟+欠占用；1024×5120 也只 300GB/s。
2. **in-graph 干扰损失仅 1.74ms（8%）**：21.73 vs 19.99，L2 污染+DRAM 状态，难以直接优化。
3. 结论："GEMV 带宽追回 5ms"不成立，可行动空间是小 shape 融合（~0.75ms）。

### 15.2 P7-2 成对 GEMV 融合（`_qwen38_p7_pair.py`）

**洞察**：模型存在天然的同输入成对调用——每层 `in_proj_b(x)→in_proj_a(x)`（48 对 48×5120）与 `k_proj(x)→v_proj(x)`（16 对 1024×5120，GQA）。合并后单 launch 行数翻倍（96/2048），占满度×2，launch 数 -64/步。

**实现**（纯 Python patch，零 kernel 改动）：
- 权重 D2H→H2D 拼进连续 buffer（行独立计算 → 与两次 M 行 launch **逐位相同**）；
- 首成员（b/k）T=1 无条件算整组 M_total 行（无跨步缓存，graph replay 安全）；次成员（a/v）校验 `data_ptr` 一致后取行切片；异常调用序 fallback 偏移指针独立算；
- T>1（prefill）走原 super() 路径，次成员 d_p/d_s 天然是行偏移指针。

**验收**（`_qwen38_p7_pair_acc.py`）：
- 63 token 逐位 **PASS**（64/64 含首 token）；验收脚本首版 FAIL @45 的教训：Run2 重 capture 后漏了 `snap/restore`——GraphDecoder capture 的 warmup 会污染 DeltaNet 状态，**每次重 capture 后必须 restore**；
- **35.3 tok/s（28.32ms/步），+4.2%，达成 35 tok/s 目标**；实际省 1.15ms/步 > 理论 0.56ms（小 shape 融合 + graph replay 每节点开销 ~5μs × 64）。

**优化史**：baseline 24.8 → P5-B 28.8 → P5-A 32.2 → P6 33.9 → **P7 35.3 tok/s**（+42%）。

### 15.3 剩余空间（28.32ms/步）

- GEMV 仍 ~19.2ms/步（隔离极限 19.99 - 融合收益）：in-graph 干扰 1.74ms 理论天花板；
- RoPE 融合进 attn kernel（~0.8ms）、MLP silu·mul（~0.5ms）、residual add 链（~0.8ms）——合计 ~2ms，36-37 tok/s 潜力；
- DN 融合 3.7ms（dn_conv+dn_rec）下探空间未剖析。

## 阶段 16：P8 多轮压力测试 — 显存台账 + OOM 阈值（2026-08-28）

### 16.1 S1-S4：台账、斜率、泄漏、LRU 容量（`_qwen38_p8_stress.py`）

| 指标 | MAXLEN=3072 | MAXLEN=6144 |
|---|---|---|
| 加载后 smi | 14825 MiB | 14817 MiB |
| prefill 后 alloc / smi | 540 / 15457 MiB | 760 / 15709 MiB |
| decode 态峰值 smi | 15565 MiB | 15777 MiB |
| 10 轮 decode 显存漂移 | **+0 MiB（无泄漏）** | **+0 MiB（无泄漏）** |
| LRU 前缀快照（CPU RAM） | 166-168 MiB/套（L≈600） | 同左，GPU 无增量 |

- **S2 斜率**：torch alloc 口径 5-7 KiB/token（静态 cache 预分配后近乎零增长）；smi 口径 KV 增量 ≈ 0.21 MiB/token（含 16 层 fp16 KV + DeltaNet 状态）。
- **S3**：10 轮 poison → 重 prefill → 63 tok decode，`first` token 与 alloc 完全一致——**无泄漏、无碎片化**，多轮长对话显存稳定。
- **S4**：LRU 快照驻留 CPU RAM，GPU 零增量——1.51GB 剩余显存与 RAG embed 模型（~100MB）共存无冲突。

### 16.2 发现 67：OOM 阈值前先撞 kernel 编译期上限

- MAXLEN=4096 首测在 decode capture 阶段 illegal memory access——**非 OOM**。根因：attn decode kernel `__shared__ float s_sc[MAXLEN]` 与 KV stride 均按编译期 `MAXLEN=3072` 锁死，L=4025 时 `s_sc[3072..4024]` 越界写共享内存。sticky error 在下一个带错误检查的 launch（layer1 in_proj_b GEMV）处冒头，一度误导定位；`CUDA_LAUNCH_BLOCKING=1` 复跑才浮到真实位置。
- **修复**：`.cu` 加 `#ifndef MAXLEN` 保护，`-DMAXLEN=6144` 独立编译 `nvfp4_attn_decode_6144.dll`（launcher 有 `max_len != MAXLEN → -2` 防线，传参即生效）；根目录写保护限制下用本地复刻版 patch（`_qwen38_p8_attnml.py`）。
- **6144 全阶段 PASS**：smi 峰值 16129 MiB，距 16GB 仅 ~255 MiB。
- **OOM 阈值裁决**（逐档实测）：3072 / 6144 / 7168 / **8192 全部 PASS**（8192 峰值 smi 16233 MiB），**9216 OOM**（S3 decode 阶段差 25 MiB：申请 108 MiB 仅剩 83 MiB，torch 总容量 15.86 GiB 含 WDDM 保留）。**有效上下文上限 ≈ 8192 tok**；attn kernel 共享内存维度理论上限 ~12k（s_sc 48KB 封顶），显存是绑定约束。
- 各档稳定态：prefill 后 alloc 540/760/864/945/1035 MiB（3072→9216），torch 口径每 token 5-14 KiB；LRU 前缀快照恒 166-168 MiB/套（CPU RAM），GPU 零增量。
- **RAG 挂载容量预算**（结论）：decode 生产配置建议 MAXLEN=6144（留 ~500 MiB 给 embed 模型 + 余量）；MAXLEN=8192 时无 RAG 空间。多前缀 LRU 数量只受 CPU RAM 限制（每套 ~167 MiB，64GB RAM 可挂数百套）。

### 16.3 过程记录（每步测试台账）

| 步骤 | 测试 | 结果 |
|---|---|---|
| 1 | MAXLEN=3072 全流程 S1-S4 | PASS，无泄漏 |
| 2 | MAXLEN=4096 | FAIL：attn kernel `s_sc[3072]` 越界（发现67，非法访问，非 OOM） |
| 3 | `.cu` 加 `#ifndef MAXLEN` + 编译 6144 DLL | PASS |
| 4 | MAXLEN=6144 全流程 | PASS，smi 峰值 16129 |
| 5 | 编译 7168 DLL + 全流程 | PASS，smi 峰值 16069 |
| 6 | 编译 8192 DLL + 全流程 | PASS，smi 峰值 16233 |
| 7 | 编译 9216 DLL + 全流程 | **OOM**（decode 阶段，差 25 MiB）→ 爆点区间 [8192, 9216) |


## 阶段 17：RAG 知识库挂载 — 14 库检索 + NVFP4 生成端到端（2026-08-28，✅ 6/6 PASS）

### 17.1 前置：知识库检索管线重建（方案 B，全量重编码）

- **起因**：14 库 15.9 万条向量由 bge-large-zh-v1.5 编码，但模型文件已丢（`D:\models` 目录消失），存量向量成"孤儿向量"。
- **方案 B 落地**：Qwen3-Embedding-0.6B（F 盘在盘，1.19GB，1024 维）全量重编码 14 库 —— beethoven 18508 条 3.9 分钟验证质量（0.87 区分度）后推 13 库，总 15.9 万条 25.9 分钟零失败。新向量行 `qwen3-emb-0.6b`，bge 旧行保留只读。
- **CPU 检索入口**（`vf_kb_router_cpu.py`）：编码器 fp32 常驻 CPU（~2.4GB RAM，显存零竞争，精度比 GPU fp16 还高一档）；首查 15s（含加载），热查询 **0.5-0.9s**。
- 统一入口 `vf_kb_router.py`：`kb_search(query)` 关键词自动路由 / `soul=` 显式指定，14 库路由终测 6/6 PASS。

### 17.2 RAG 三段式挂载（`_qwen38_rag.py`）

```
[检索] CPU: Qwen3-Embedding-0.6B fp32 → kb_search(top_k=4, score≥0.35)
[组装] 命中条目 → RAG 前缀 ≤1500 字（entry_id + 相似度标注）
[生成] GPU: NVFP4 Qwen3.8（MAXLEN=3072, chat template, GraphDecoder 35 tok/s）
```

- 多轮流程（p4a 模式）：每题 poison_cache → prefill → refresh_deltanet_state → set_pos(L) → decode。
- 验收 6 题（einstein×2 / strategy×2 / beethoven×2）：**全部生成高质量引用答案**——隧穿透射系数公式与 STM 应用、纳什 1950 博士论文/1994 诺奖出处、三全音替代 Db7↔G7、奏鸣三部结构；"先发优势"一题正确表现"知识库未直接提供→如实说明"的诚实行为。
- 性能：生成 32.1-33.6 tok/s（RAG 前缀 ~700 tok 使 L 变长，较纯对话 35.3 略降，符合预期）；检索热查询 <1s；端到端单题 ~5s。

### 17.3 发现 68：三连环坑（patch 时序 / chat template / MAXLEN stride）

1. **patch 前必须先 prefill**：`patch_fused_deltanet` 从 cache 抓 conv/recurrent 状态，StaticCache 未跑过前向时 `conv_states=None` 直接 AttributeError。引导 prefill 一次即可。
2. **裸文本 prompt 会退化**：Qwen3.8 不走 `apply_chat_template` 时输出退化为"!!!!!"循环（首 token 正常，因来自 prefill logits）；必须 chat template + `enable_thinking=False`。
3. **attn kernel stride 绑定**：`patch_attn_decode` 硬编码加载 3072-stride DLL，cache 用 6144 时 KV 寻址错位 → 首步起 decode 全烂（症状与 2 叠加，先修 2 才暴露 3）。RAG 场景编码器在 CPU 无显存预留需求，MAXLEN=3072 走默认 DLL 即可；需 6144 时须换 `nvfp4_attn_decode_6144.dll`（根目录写保护，本地 patch 方案见阶段 16 `_qwen38_p8_attnml.py`）。
- 教训：**"管线跑通"≠"语义正确"——性能指标全正常（tok/s、检索命中）时输出仍可能是乱码，验收必须含人读语义质量项**。




---

## 阶段 18：P9 RoPE QK + MLP silu·mul 融合 — 验证性优化收尾（2026-08-29，✅ 逐位 PASS，提速 +0.4% 裁决）

### 18.1 设计（`fp4研究\nvfp4_rope_silu.cu`，sm_70）

- **动机**（P6 profile 预估）：decode 每步 elementwise 残余 ~2.0ms——16 层 full_attention 的 `apply_rotary_pos_emb` torch 链 ~10 kernel/层 (~0.8ms/步) + 64 层 MLP `silu(gate)*up` 2 kernel/层 (~0.5ms/步)。
- **两个融合 kernel**：
  1. `rope_qk_kernel`：grid=(hq+hkv) blocks × 256 threads，每 block 一个头；T=1 时 q (24×256) + k (4×256) 一次 launch 替换整条 torch 链（chunk/cat/mul/add/广播）
  2. `silu_mul_kernel`：grid=ceil(n/256)，`silu(g)·u` 单 launch 替换 2 kernel/层
- **位精确舍入链**（逐 op 对齐 torch CUDA opmath=fp32）：
  - RoPE：`t1=f16(q·cos)` `t2=f16(rot(q)·sin)` `out=f16(t1+t2)`——rotate_half=(-x2,x1) 符号翻转位精确
  - silu：`s=f16(g/(1+expf(-g)))` 必须用**除法**（乘倒数有 ulp 差异）；`out=f16(f32(s)·f32(u))`
  - `expf` 软件精确版，禁用 `__expf`（发现 64 教训）
- **注入方式**：RoPE 走模块级 monkey-patch（`transformers.models.qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb`）；MLP 替换 `Qwen3_5MLP.forward` 的 T==1 分支。T>1（prefill/MTP）回退原路径。
- **融合条件守卫**：unsqueeze_dim==1、T==1、head_dim==256、每头连续布局（stride 校验）、fp16、全旋转（rotary_dim==head_dim）——不满足全部走原路径。

### 18.2 结果（`fp4研究\_qwen38_p9_acc.py`，三次运行）

| 运行 | Run1 baseline (P5-A+P6+P7) | Run2 P9 | 增益 | 63 token 逐位 | 人读 |
|---|---|---|---|---|---|
| 首跑 | 33.6 tok/s | 35.5 tok/s | +5.6% | PASS 64/64 | — |
| 复跑 | 35.3 tok/s | 35.4 tok/s | +0.4% | PASS 64/64 | PASS |
| 三跑 | 35.3 tok/s | 35.5 tok/s | +0.4% | PASS 64/64 | PASS |

- 数值无损：三次逐位 PASS，融合 kernel 与 torch 链 bit-exact。
- **诚实裁决：真实稳态增益只有 +0.4%（35.3→35.5 tok/s），36-37 tok/s 目标未达成**。首跑 +5.6% 是 GPU 冷启动预热效应。
- **根因（发现 69）**：CUDA Graph replay 下 launch 开销本已消除，elementwise kernel 纯执行时间只剩 ~0.1-0.3ms/步——P6 profile 预估的 ~2ms 含 launch 开销，融合空间被 graph 提前吃掉。当前瓶颈主体是 497 个 NVFP4 GEMV 本身（~25ms/步）。

### 18.3 发现 69：CUDA Graph 场景下 elementwise 融合收益衰减一个数量级

- eager 模式下"消灭 launch 数"是 elementwise 融合的主要收益来源；graph replay 把整步 decode 的 kernel 序列固化重放，launch 开销≈0，融合只剩"减少 kernel 纯执行时间与中间访存"的小头。
- 量化预估：profile 得到的 per-op 时间含 launch，graph 下直接引用会**高估 elementwise 融合收益约一个数量级**（本项目 2.0ms 预估 → 实际 ~0.1ms）。
- 后续优化裁决原则：graph 内 kernel 时间必须用 CUDA event/graph 内分解测，不能拿 eager profile 数字做融合决策。

### 18.4 工程坑（build 环境）

1. **bat 换行符陷阱**：Write 工具产出的 bat 是 LF，cmd 按 CRLF 解析会把每行首字符吃掉（`'gram' is not recognized`）→ 写后必须转 CRLF。
2. **vcvars64 静默失效**：`call vcvars64.bat >nul 2>&1` 后 nvcc 仍不在 PATH（BuildTools 环境变量注册路径与实际不符）→ 用项目已验证方式：CUDA 12.8 nvcc 全路径 + `-ccbin` 显式指定 MSVC cl.exe 目录（同 `nvfp4_perf_compare.py`）。
3. **kernel launch 必须绑定传入 stream**：Python 侧传了 `cuda_stream` 但 `<<<>>>` 没用（默认流）→ CUDA Graph capture 时默认流操作会炸或漏出 graph；本 DLL 首版即犯，编译前修正为 `<<<grid, block, 0, st>>>`。

### 18.5 阶段 18 产出

- 生产变更：无（P9 定位为验证性优化，kernel 正确但收益不达预期；`_qwen38_p9_fuse.py` 保留可挂载，默认不入主链）
- 工具与内核：`nvfp4_rope_silu.cu/.dll`（stream 绑定修正版）、`_qwen38_p9_fuse.py`（patch 模块）、`_qwen38_p9_acc.py`（验收脚本，含人读门槛）、`_build_p9.bat`（CRLF + 全路径 nvcc）
- 知识入库：cezanne.db 新增 3 条（kernel 位精确设计 / graph 融合收益衰减裁决 / build 工程坑）
- 当前生产栈不变：R1+R2 + v8 GEMV + P7 成对融合 = **35.3 tok/s** decode

---

## 阶段 11：P0 视觉塔实装 + P1 v8 E2E 验收 + P2 三项定论（2026-08-17）

### 11.1 P0 — M1 视觉塔端到端集成（✅ 上半场完成）

- 打包：`_p2_pack_vision.py` → `vision_m1.npz` 542MB（90 个深层 INT8 per-row Linear + 20 个浅层 FP16）
- 集成：`_qwen38_infer.py` 新增 `VLinear`（INT8 常驻 + 前向反量化 i8.f32×scale.f32→fp16）、`load_vision_m1()` 懒加载、`encode_image()/splice_image_embeds()/build_mm_pos4()`、GraphDecoder 支持 mrope 4D 逻辑位置
- 验收：`_p2_vision_e2e.py` 视觉编码 cos=0.99988 / 行最低 0.9991；多模态 prefill logits 一致；文本路径 96 token 全一致无回退
- 知识入库 nvfp4-025~029（上批），本批续 036~043

### 11.2 P1 — v8 GEMV 端到端验收（✅ 超额达标）

| 路径 | v1 (LUT) | v8 (PRMT) | 提升 |
|---|---|---|---|
| graph R1 | 21.35 tok/s | 25.69 tok/s | +20.3% |
| **graph R1+R2** | **25.92 tok/s** | **32.02 tok/s** | **+23.5%** |

- 内核带宽 +53% 被非 GEMV 环节稀释为端到端 +23.5%（Amdahl）
- v8 与 v1 hfma2 链长不同（16 vs 8 元素/链）非 bit-exact，但同 prompt 生成**逐字一致**；graph/eager 内部 63/63
- 切换：`env VF_GEMV=v1|v8`（默认 v8）+ `set_gemv_t1()`；v8 为生产默认

### 11.3 P2-1 — 3-stage 缓冲 TC 内核复验（✅ 定论：不采纳）

- 实现：v5 主循环原为硬编码双缓冲（`rd^=1`），`-DNUM_STAGES=3` 下真实现环形流水（LDG 前瞻 2 tile、regs_a/b/c 轮转、k%3 环形 buffer）；shared 27.0→40.5KB、+9 regs（occupancy 不降）
- 正确性：6 case 全部 **bit-exact** + 真值 PASS
- 性能（gpu_sync 计时）：平均 **-1.4%**（7/8 尺寸 -2~-3%，仅 4096×4096×4992 +6.4%）
- 机制：GEMV 计算轻→前瞻有效（+14.1%）；TC 计算密集→单 tile compute 已覆盖 LDG 延迟，V100 无 cp.async 寄存器驻留反成负担
- 附带发现两个工程坑：**launch 8 参 UB**（ctypes 少传 stream 尾参=读栈垃圾，历史 test 脚本"碰巧能跑"）；**bench 必须 gpu_sync**（异步发射 wall-clock 只测 ~3μs 发射时间，历史 v5_test 绝对耗时存疑；真实 v5 吞吐 22-33 TF）

### 11.4 P2-2 — MTP 多步投机（✅ 定论：关闭）

- NVFP4 draft：接受率 58%（T1 53%/T2 61%），22.5 tok/s
- 保精度重测：FP16 draft 542MB **OOM**；Q8 per-row draft（275MB + 256 行/块分块 f16 反量化，临时峰值 ~5MB）接受率 **62%** → 仅 +4pp，瓶颈=单层 MTP 头预测上限（~60%）非量化损伤，远低于 87% 阈值
- 端到端倒挂：22.5 < 32.02 tok/s（v2 verify 54ms/轮 + 回滚开销；收支平衡需接受率 ~85%+）
- greedy 分歧两版本同在 @91 → 链式语义边界问题，与量化无关，方向关闭不深挖

### 11.5 P2-3 — mma PTX 内联（✅ 定论：物理不可行，关闭）

- ptxas 编译实证（CUDA 12.8，-arch=sm_70，探针 `_p2_mma_probe.cu`）：
  - `mma.m16n8k8` → **requires sm_75+**（Turing）
  - `mma.m16n8k16` → **requires sm_80+**（Ampere）
- 架构事实：Volta TensorCore 唯一接口 = wmma API；PTX mma 自 Turing 引入；Volta 亦无 ldmatrix。"+20-40%" 文献结论来自 Ampere+，Volta 不适用
- 结论：v5 wmma 即 V100 最优；mma 优化属未来 sm_80+ 硬件议题
- 踩坑：BOM 陷阱复发（中文注释 .cu 无 BOM → MSVC GBK 解析错位报假错）；PowerShell `Select-Object -First N` 截断 stderr 掩盖 ptxas error（判成败必须看 exit code）

### 11.6 阶段 11 产出

- 生产变更：`_qwen38_infer.py`（视觉塔 + v8 默认）、`nvfp4_cuda_v5.cu`（NUM_STAGES 参数化 + gpu_sync 导出，`#else` 分支与原版逐字符一致，重编后冒烟 PASS）、`_qwen38_mtp.py`（Q8Linear draft 模式）
- 工具脚本：`_p2_stage3_ab.py`（同源编译对照）、`_p2_mma_probe.cu`（sm_70 mma 可行性证据）、`_p2_v5_smoke.py`
- 知识入库：cezanne.db **11368 → 11376**（nvfp4-036~043：基准 1 + 定论 3 + 工程 1 + 坑 3），F 盘镜像同步完成
- 当前生产栈：R1+R2 + v8 GEMV = **32.02 tok/s** decode，多模态视觉塔 M1 就绪
