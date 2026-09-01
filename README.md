# NVFP4 on V100 — Software-Simulated FP4 Inference

**Running NVFP4 (E2M1 + E8M0 block-scaled FP4) inference on a GPU that has no FP4 hardware — a V100 16GB — with bit-exact output and 35.5 tok/s on a 27B model.**

[中文说明](README.zh.md)

---

## What is this?

NVFP4 is NVIDIA's 4-bit float format (E2M1 payloads + E8M0 per-block scales, 0.5625 bytes/param) designed for Blackwell's FP4 tensor cores. A V100 (sm_70) has none of that hardware. This repository is the complete research record of doing it anyway — **in software** — and making it fast:

- A full quantization pipeline that packs Qwen-family HF weights into an NVFP4 container (`nvfp4_packed.bin` + index).
- A family of custom CUDA kernels that decode-and-compute NVFP4 directly on V100, including a GEMV that exploits the PRMT byte-permute instruction as a hardware lookup table for E2M1 decoding, reaching **792 GB/s** effective bandwidth.
- Monkey-patch integration into HuggingFace `transformers` (static KV cache, fused RoPE / RMSNorm / silu·mul / attention-decode / DeltaNet kernels, CUDA Graph capture of the whole decode step).
- An honest engineering ledger: **69 numbered findings**, silent-bug post-mortems, and an 18-stage experiment log.

**Headline results** (Qwen3.8-27B, V100 16GB, 14.4GB weights resident):

| Metric | Value |
|---|---|
| Memory density | 0.5625 B/param |
| Numerical fidelity | bit-exact vs FP16 baseline over 63-token generations |
| Decode speed | 35.5 tok/s (production stack R1+R2 + v8 GEMV + paired fusion) |
| GEMV bandwidth | 792 GB/s effective (PRMT LUT decode) |
| Decode step anatomy | 497 NVFP4 GEMVs ≈ 25 ms/step |

## Repository Layout

```
docs/          Experiment reports (Chinese): 18-stage full study, pitfall retrospective,
               Blackwell-gap analysis — the authoritative narrative
codec/         NVFP4 codec: E2M1/E8M0 encode/decode in pure Python (+ fast variant)
kernels/       All CUDA kernel sources, v1 → v8 evolution:
                 nvfp4_cuda.cu .. v5.cu    GEMV iterations
                 nvfp4_gemv.cu             v8 PRMT hardware-LUT decode GEMV
                 nvfp4_baseline_f16.cu     FP16 reference baseline
                 nvfp4_rmsnorm.cu / nvfp4_attn_decode.cu / nvfp4_rope_silu.cu
                 nvfp4_dn_fused.cu         DeltaNet fused kernel
               (build_p9.bat shows the nvcc + MSVC toolchain invocation)
simulation/    Bit-level software simulation & kernel microbenchmarks
pipeline/      _qwen38_nvfp4_pack.py  HF weights → NVFP4 packed container
               _qwen38_infer.py       skeleton + weight upload + decode engine bridge
               _qwen38_r1.py / r2.py  baseline monkey-patch layers
experiments/   Stage-by-stage scripts (P4a → P9), kept with original names so they
               cross-reference the experiment ledger in docs/; data/ holds bench JSONs
```

## Getting Started

Prerequisites: Windows, CUDA 12.8 toolkit + MSVC (for `nvcc`), PyTorch with CUDA, a V100-class GPU (16GB), and Qwen model weights on local disk.

```bash
# 1) Compile kernels (adjust NVCC/CCBIN paths in the bat to your machine)
cd kernels && build_p9.bat

# 2) Pack your model into NVFP4 (weights NOT included in this repo)
set NVFP4_MODEL_DIR=F:\models\Qwen3.8-27B
python pipeline\_qwen38_nvfp4_pack.py

# 3) Run the acceptance chain (bit-exactness + speed)
python experiments\_qwen38_p9_acc.py
```

`NVFP4_MODEL_DIR` (environment variable) selects the model directory; all DLL lookups are repository-relative.

## The Interesting Findings

The full story lives in `docs/`. Highlights that generalize beyond this project:

- **A "silent bug" taxonomy**: FP4 pipelines don't crash when they're wrong — answers just quietly become garbage. Three chained silent bugs (exp2f precision loss, cross-warp reduction loss, RoPE/template/stride mismatch) are dissected with repro and fix.
- **Finding 69**: under CUDA Graph replay, kernel-launch overhead disappears, so elementwise-op fusion gains decay by an order of magnitude (expected +5.6%, measured +0.4%). Profile before fusing.
- **Compile-time-locked shared arrays**: `s_sc[MAXLEN]` must be fixed at DLL compile time — the "illegal access" you get from a larger MAXLEN is not OOM.
- **The honest-miss philosophy**: acceptance is 63-token bit-exact comparison against the FP16 baseline, not "looks reasonable".

## Status & Scope

This is research code published for reproducibility and reference: script-style modules, Chinese comments, Windows-centric paths, and experiment scripts that assume the exact hardware/software stack described in `docs/`. RAG integration (built on private knowledge bases) is intentionally not included. Model weights are not included — derive them with the packing pipeline.

## License

[MIT](LICENSE)
