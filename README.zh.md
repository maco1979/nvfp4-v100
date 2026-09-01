# NVFP4 V100 软件模拟推理

**在没有 FP4 硬件的 GPU（V100 16GB）上跑 NVFP4（E2M1 + E8M0 块缩放 FP4）——输出与 FP16 基线逐位一致，27B 模型 35.5 tok/s。**

[English](README.md)

---

## 这是什么？

NVFP4 是 NVIDIA 的 4bit 浮点格式（E2M1 载荷 + E8M0 块缩放，0.5625 字节/参数），为 Blackwell 的 FP4 张量核心设计。V100（sm_70）完全没有这些硬件。本仓库是"偏要用软件模拟跑起来"的完整研究记录：

- 完整量化流水线：Qwen 系 HF 权重 → NVFP4 打包容器（`nvfp4_packed.bin` + 索引）。
- 一族自研 CUDA kernel，直接在 V100 上解码并计算 NVFP4——其中 GEMV 用 PRMT 字节重排指令当 E2M1 解码硬件查表，实测有效带宽 **792 GB/s**。
- 通过 monkey-patch 接入 HuggingFace `transformers`（静态 KV cache、RoPE / RMSNorm / silu·mul / attention-decode / DeltaNet 融合 kernel、整步 decode 的 CUDA Graph 捕获）。
- 一本诚实的工程台账：**69 个编号发现**、静默 Bug 尸检、18 个阶段的实验记录。

**核心数字**（Qwen3.8-27B，V100 16GB，权重常驻 14.4GB）：

| 指标 | 数值 |
|---|---|
| 显存密度 | 0.5625 B/param |
| 数值保真 | 63 token 生成与 FP16 基线逐位一致 |
| 生成速度 | 35.5 tok/s（生产栈 R1+R2 + v8 GEMV + 成对融合） |
| GEMV 带宽 | 792 GB/s 有效带宽（PRMT 查表解码） |
| 解剖数据 | 每步 497 个 NVFP4 GEMV ≈ 25 ms/step |

## 仓库结构

```
docs/          实验报告（中文）：18 阶段全量整理、坑点复盘、
               Blackwell 差距分析——权威叙事
codec/         NVFP4 编解码器：E2M1/E8M0 纯 Python 实现（+ fast 版）
kernels/       全部 CUDA kernel 源码，v1 → v8 演进：
                 nvfp4_cuda.cu .. v5.cu    GEMV 迭代
                 nvfp4_gemv.cu             v8 PRMT 硬件查表解码 GEMV
                 nvfp4_baseline_f16.cu     FP16 参考基线
                 nvfp4_rmsnorm.cu / nvfp4_attn_decode.cu / nvfp4_rope_silu.cu
                 nvfp4_dn_fused.cu         DeltaNet 融合 kernel
               （build_p9.bat 给出 nvcc + MSVC 工具链调用方式）
simulation/    位级软件模拟 + kernel 微基准
pipeline/      _qwen38_nvfp4_pack.py  HF 权重 → NVFP4 打包容器
               _qwen38_infer.py       骨架 + 权重上传 + 解码引擎桥接
               _qwen38_r1.py / r2.py  基础 monkey-patch 层
experiments/   阶段实验脚本（P4a → P9），保留原文件名以便与 docs/ 里的
               实验台账互相索引；data/ 存基准测试 JSON
```

## 快速开始

前置条件：Windows、CUDA 12.8 工具链 + MSVC（`nvcc` 编译用）、CUDA 版 PyTorch、V100 级显卡（16GB）、本地 Qwen 模型权重。

```bash
# 1) 编译 kernel（bat 里的 NVCC/CCBIN 路径按你机器调整）
cd kernels && build_p9.bat

# 2) 把模型打包成 NVFP4（本仓库不含权重）
set NVFP4_MODEL_DIR=F:\models\Qwen3.8-27B
python pipeline\_qwen38_nvfp4_pack.py

# 3) 跑验收链（逐位一致性 + 速度）
python experiments\_qwen38_p9_acc.py
```

`NVFP4_MODEL_DIR`（环境变量）指定模型目录；所有 DLL 查找均为仓库相对路径。

## 值得一看的发现

完整故事在 `docs/`。几条超出本项目本身的通用结论：

- **"静默 Bug"分类学**：FP4 流水线出错不会崩溃——回答只会悄悄变垃圾。三连环静默 Bug（exp2f 精度丢失、跨 warp 归约丢失、RoPE/模板/stride 错位）附复现与修复。
- **发现 69**：CUDA Graph 重放下 kernel 启动开销消失，elementwise 算子融合收益衰减一个数量级（预期 +5.6%，实测 +0.4%）。先 profile 再融合。
- **编译期锁定的共享数组**：`s_sc[MAXLEN]` 必须在 DLL 编译期定死——调大 MAXLEN 报的 "illegal access" 不是 OOM。
- **诚实验收哲学**：验收标准是与 FP16 基线 63 token 逐位比对，不是"看起来还行"。

## 状态与范围

这是为可复现性和参考发布的研究代码：脚本式模块、中文注释、Windows 中心路径，实验脚本假定 docs/ 里描述的软硬件栈。RAG 集成（基于私有知识库）有意不包含。模型权重不含——用量化流水线自行生成。

## 许可证

[MIT](LICENSE)
