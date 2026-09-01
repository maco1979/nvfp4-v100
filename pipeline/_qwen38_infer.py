# -*- coding: utf-8 -*-
"""Qwen3.8-27B NVFP4 端到端推理桥接 (阶段B/C)
=================================================
- nn.Linear -> QLinear: 权重 NVFP4 常驻 GPU (0.5625 B/param)
  - T=1 (decode): 自研 GEMV kernel (nvfp4_gemv.dll, 305-392 GB/s)
  - T>1 (prefill): v5 TensorCore GEMM (nvfp4_cuda_v5.dll)
- 非 Linear 参数 (norm/bias/1D/embed): fp16 GPU
- meta 骨架 -> patch Linear -> to_empty -> 填充 -> rope 重建
- decode 默认 R1+R2 组合快速路径 (StaticCache + CUDA Graph + fused DeltaNet, ~21.8 tok/s)
用法: python _qwen38_infer.py "你的问题" [生成token数=128] [max_cache_len=1024]
"""
import ctypes
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.abspath(__file__))
KDIR = os.path.join(os.path.dirname(BASE), "kernels")
MODEL_DIR = os.environ.get("NVFP4_MODEL_DIR", r"F:\models\Qwen3.8-27B")
BIN = os.path.join(MODEL_DIR, "nvfp4_packed.bin")
IDX = os.path.join(MODEL_DIR, "nvfp4_index.json")
EMB = os.path.join(MODEL_DIR, "fp16_embed.npy")
EMB_P = os.path.join(MODEL_DIR, "nvfp4_embed_packed.npy")    # C1: embed NVFP4
EMB_S = os.path.join(MODEL_DIR, "nvfp4_embed_scales.npy")
MISC = os.path.join(MODEL_DIR, "fp32_misc.npz")
VISION_NPZ = os.path.join(MODEL_DIR, "vision_m1.npz")   # P0: M1 视觉塔 (浅FP16+深INT8)


def load_gemv():
    lib = ctypes.CDLL(os.path.join(KDIR, "nvfp4_gemv.dll"))
    lib.gpu_malloc.argtypes = [ctypes.c_size_t]
    lib.gpu_malloc.restype = ctypes.c_void_p
    lib.gpu_free.argtypes = [ctypes.c_void_p]
    lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_h2d.restype = ctypes.c_int
    lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.gpu_memcpy_d2h.restype = ctypes.c_int
    lib.launch_nvfp4_gemv.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.launch_nvfp4_gemv.restype = ctypes.c_int
    lib.launch_nvfp4_gemv_v8.argtypes = [          # P1: PRMT 硬件查表 792GB/s
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.launch_nvfp4_gemv_v8.restype = ctypes.c_int
    lib.launch_nvfp4_gemv2.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.launch_nvfp4_gemv2.restype = ctypes.c_int
    return lib


def load_v5():
    lib = ctypes.CDLL(os.path.join(KDIR, "nvfp4_cuda_v5.dll"))
    lib.launch_nvfp4_gemmtc_v5.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
        ctypes.c_int, ctypes.c_int, ctypes.c_int]
    lib.launch_nvfp4_gemmtc_v5.restype = ctypes.c_int
    return lib


GEMV = load_gemv()
V5 = load_v5()
H2D = GEMV.gpu_memcpy_h2d
BIN_FD = open(BIN, "rb")
UPLOAD_BYTES = [0]
# P1: T=1 GEMV 内核选择 (env VF_GEMV=v1|v8, 默认 v8 PRMT 792GB/s)
# 注意: v8 与 v1 hfma2 链长不同 (16 vs 8 元素/链) — 非 bit-exact, 切换需重验 token
GEMV_T1 = {"v1": GEMV.launch_nvfp4_gemv,
           "v8": GEMV.launch_nvfp4_gemv_v8}.get(
               os.environ.get("VF_GEMV", "v8"), GEMV.launch_nvfp4_gemv_v8)


def set_gemv_t1(name):
    """运行时切换 T=1 内核 (A/B 对照用): 'v1' 或 'v8'"""
    global GEMV_T1
    fn = {"v1": GEMV.launch_nvfp4_gemv,
          "v8": GEMV.launch_nvfp4_gemv_v8}[name]
    GEMV_T1 = fn
    return fn


def upload_nvfp4(entry):
    d_p = GEMV.gpu_malloc(entry["packed_bytes"])
    d_s = GEMV.gpu_malloc(entry["scales_bytes"])
    assert d_p and d_s, "gpu_malloc failed"
    BIN_FD.seek(entry["packed_off"])
    pb = BIN_FD.read(entry["packed_bytes"])
    BIN_FD.seek(entry["scales_off"])
    sb = BIN_FD.read(entry["scales_bytes"])
    assert H2D(d_p, pb, len(pb)) == 0 and H2D(d_s, sb, len(sb)) == 0
    UPLOAD_BYTES[0] += len(pb) + len(sb)
    return d_p, d_s


class PackedEmb:
    """C1: NVFP4 packed embed 查表代理 — 接口对齐 weight_cpu (shape + [idx] -> fp16 tensor)
    mmap 驻留 (RAM 0.68GB vs fp16 2.37GB); 行解码 µs 级 (LUT×exp2 向量化)"""
    CP = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=np.float32)

    def __init__(self, path_p, path_s):
        self.packed = np.load(path_p, mmap_mode="r")       # (V, H/2) uint8
        self.scales = np.load(path_s, mmap_mode="r")       # (V, H/16) uint8
        V, Hh = self.packed.shape
        self.shape = (V, Hh * 2)
        lut = np.where(np.arange(16) & 8, -1.0, 1.0) * self.CP[np.arange(16) & 7]
        self.lut = lut.astype(np.float32)

    def __getitem__(self, idx):
        scalar = np.isscalar(idx) or (torch.is_tensor(idx) and idx.ndim == 0)
        idx = np.asarray(idx.cpu() if torch.is_tensor(idx) else idx).reshape(-1)
        p, sb = self.packed[idx], self.scales[idx]
        n = p.shape[0]
        nib = np.empty((n, p.shape[1] * 2), dtype=np.uint8)
        nib[:, 0::2] = p & 0x0F
        nib[:, 1::2] = p >> 4
        w = self.lut[nib].reshape(n, -1, 16)
        sc = np.exp2(sb.astype(np.float64) - 127.0)        # 铁律: scale f64 中间
        w = (w * sc[:, :, None]).astype(np.float16).reshape(n, -1)
        return torch.from_numpy(w[0] if scalar else w)


class QEmbedding(nn.Module):
    """embed 留 CPU RAM (省 2.5GB VRAM), 每步只传查到的行 (~10KB)"""
    def __init__(self, weight_cpu):
        super().__init__()
        self.weight_cpu = weight_cpu      # (V, H) fp16 CPU tensor, 普通属性防 to_empty 清空
        self.num_embeddings, self.embedding_dim = weight_cpu.shape

    def forward(self, ids):
        idx = ids.detach().cpu().reshape(-1)
        rows = self.weight_cpu[idx]                    # CPU gather
        return rows.to("cuda", dtype=rows.dtype).view(*ids.shape, -1)


class QLinear(nn.Module):
    """权重 NVFP4 常驻 GPU 的 Linear 替身 (forward-only)"""

    def __init__(self, d_p, d_s, M, K, bias=None):
        super().__init__()
        self.d_p, self.d_s, self.M, self.K = d_p, d_s, M, K
        self.bias = bias

    def forward(self, x):
        K = self.K
        lead = x.shape[:-1]
        x2 = x.reshape(-1, K).contiguous()
        if x2.dtype != torch.float16:
            x2 = x2.to(torch.float16)
        T = x2.shape[0]
        stream = torch.cuda.current_stream().cuda_stream
        if T == 1:
            out = torch.empty((1, self.M), dtype=torch.float16, device=x.device)
            rc = GEMV_T1(self.d_p, self.d_s,
                         x2.data_ptr(), out.data_ptr(),
                         self.M, K,
                         ctypes.c_void_p(stream))
            assert rc == 0, f"gemv rc={rc}"
        elif T == 2:                                 # MTP verify batch: 一次权重读 2 token
            out = torch.empty((2, self.M), dtype=torch.float16, device=x.device)
            rc = GEMV.launch_nvfp4_gemv2(self.d_p, self.d_s,
                                         x2[0].data_ptr(), x2[1].data_ptr(),
                                         out.data_ptr(), self.M, K,
                                         ctypes.c_void_p(stream))
            assert rc == 0, f"gemv2 rc={rc}"
        else:
            xt = x2.t().contiguous()                     # (K, T)
            o2 = torch.empty((self.M, T), dtype=torch.float16, device=x.device)
            rc = V5.launch_nvfp4_gemmtc_v5(self.d_p, self.d_s, xt.data_ptr(),
                                           o2.data_ptr(), self.M, T, K,
                                           ctypes.c_void_p(stream))
            assert rc == 0, f"gemm rc={rc}"
            out = o2.t().contiguous()
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*lead, self.M)


class VLinear(nn.Module):
    """P0 视觉塔 M1 Linear: 深层 INT8 per-row 常驻 + 前向反量化
    (i8→f32 × scale.f32 → fp16), 数值语义与 _p1_final.q_int8_row 严格一致;
    浅层 (blocks.0-4) 直接常驻 FP16 权重"""

    def __init__(self, bias=None, i8=None, scale=None, w_fp16=None):
        super().__init__()
        self.i8, self.scale, self.w = i8, scale, w_fp16   # 普通属性防 to_empty 清空
        self.bias = bias
        self.M, self.K = (i8.shape if i8 is not None else w_fp16.shape)

    def forward(self, x):
        w = self.w if self.w is not None else (
            self.i8.to(torch.float32) * self.scale.unsqueeze(1)).to(torch.float16)
        return torch.nn.functional.linear(x.to(torch.float16), w, self.bias)


def st_name(name):
    """骨架参数名 -> safetensors 名 (ForCausalLM 的 model.* 对应 checkpoint 的 model.language_model.*)"""
    if name.startswith("model."):
        return "model.language_model." + name[len("model."):]
    return name


def patch_linears(model, index, misc):
    """meta 阶段: nn.Linear -> QLinear (丢弃 meta 权重, 省 54GB 分配)"""
    patched = 0
    misc_keys = set(misc.files)
    for pname, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            wkey = st_name(f"{pname}.weight")
            if wkey not in index:
                print(f"  [warn] {wkey} 不在量化 index, 保留原样")
                continue
            e = index[wkey]
            d_p, d_s = upload_nvfp4(e)
            bias = None
            bkey = st_name(f"{pname}.bias")
            if module.bias is not None:
                if bkey in misc_keys:
                    bias = torch.from_numpy(misc[bkey]).to(torch.float16).cuda()
                else:
                    print(f"  [warn] {bkey} 缺失, 置零")
                    bias = torch.zeros(module.bias.shape,
                                       dtype=torch.float16, device="cuda")
            q = QLinear(d_p, d_s, e["M"], e["K"], bias=bias)
            parts = pname.split(".")
            parent = (model.get_submodule(".".join(parts[:-1]))
                      if len(parts) > 1 else model)
            setattr(parent, parts[-1], q)
            patched += 1
    return patched


def fill_rest(model, misc, embed):
    """to_empty 后: 填充剩余 parameter (norm/bias/conv/embed)"""
    filled, skipped = 0, []
    misc_keys = set(misc.files)
    for name, p in model.named_parameters():
        if "embed_tokens" in name:
            p.data.copy_(torch.from_numpy(np.asarray(embed)).to(torch.float16))
            filled += 1
            continue
        key = st_name(name)
        if key in misc_keys:
            p.data.copy_(torch.from_numpy(misc[key]).to(torch.float16).cuda())
            filled += 1
        else:
            skipped.append(key)
    return filled, skipped


def rebuild_rope(model):
    """to_empty 后重建 non-persistent rope buffer (inv_freq)
    注意: Qwen3_5 的 rope 参数存在 config.rope_parameters dict 里,
    ROPE_INIT_FUNCTIONS 认不出来 -> 直接用模型自带 rotary 类重建"""
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding
    n = 0
    for mod in model.modules():
        if isinstance(mod, Qwen3_5TextRotaryEmbedding):
            fresh = Qwen3_5TextRotaryEmbedding(mod.config, torch.device("cuda"))
            mod.inv_freq = fresh.inv_freq
            mod.original_inv_freq = fresh.original_inv_freq
            mod.attention_scaling = fresh.attention_scaling
            n += 1
    return n


# ---------------- P0: M1 视觉塔 (2026-08-17, 方案 M1 见 cezanne nvfp4-025~029) ----------------


def load_vision_m1(npz_path=VISION_NPZ):
    """P0-2: M1 视觉塔懒加载 — 深层 INT8 常驻 (~258MB) + 浅层 FP16 + 非 Linear 参数 fp16"""
    from accelerate import init_empty_weights
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5VisionModel, Qwen3_5VisionRotaryEmbedding)
    cfg_d = json.load(open(os.path.join(MODEL_DIR, "config.json"), encoding="utf-8"))
    vcfg = Qwen3_5VisionConfig.from_dict(cfg_d["vision_config"])
    with init_empty_weights():
        vision = Qwen3_5VisionModel._from_config(vcfg)
    z = np.load(npz_path)
    zkeys = set(z.files)
    n_i8 = n_f16 = 0
    for pname, module in list(vision.named_modules()):   # meta 阶段: Linear -> VLinear
        if not isinstance(module, nn.Linear):
            continue
        wkey, bkey = f"{pname}.weight", f"{pname}.bias"
        bias = (torch.from_numpy(z[bkey]).to(torch.float16).cuda()
                if bkey in zkeys else None)
        if wkey + "::i8" in zkeys:                       # 深层 INT8 per-row
            q = VLinear(bias=bias, i8=torch.from_numpy(z[wkey + "::i8"]).cuda(),
                        scale=torch.from_numpy(z[wkey + "::sc"]).cuda())
            n_i8 += 1
        else:                                            # 浅层 FP16 (M1 KEEP blocks.0-4)
            q = VLinear(bias=bias,
                        w_fp16=torch.from_numpy(z[wkey]).to(torch.float16).cuda())
            n_f16 += 1
        parts = pname.split(".")
        parent = (vision.get_submodule(".".join(parts[:-1])) if len(parts) > 1
                  else vision)
        setattr(parent, parts[-1], q)
    vision.to_empty(device="cuda")
    filled = 0
    for name, p in vision.named_parameters():            # 非 Linear: conv/norm/pos_embed
        assert name in zkeys, f"npz 缺 {name}"
        p.data = torch.from_numpy(z[name]).to(torch.float16).cuda()
        filled += 1
    for m in vision.modules():                           # 重建 non-persistent inv_freq
        if isinstance(m, Qwen3_5VisionRotaryEmbedding):
            m.inv_freq = Qwen3_5VisionRotaryEmbedding(m.dim, m.theta).cuda().inv_freq
    vision.eval()
    torch.cuda.synchronize()
    free, _t = torch.cuda.mem_get_info()
    print(f"  vision M1: INT8 {n_i8} + FP16 {n_f16} Linear | fill {filled} param | "
          f"剩余 {free/1e9:.2f}GB", flush=True)
    return vision


def encode_image(vision, pixels, grid_thw):
    """P0-2 多模态入口: pixels (N, C·t·P·P) + grid (n,3) → image_embeds (Π/merge², 5120)"""
    with torch.no_grad():
        return vision(pixels.to(torch.float16), grid_thw=grid_thw).pooler_output


def splice_image_embeds(embed_layer, ids, image_embeds, image_token_id):
    """ids 中 image_pad 段替换为 image_embeds → inputs_embeds (1,L,H)"""
    emb = embed_layer(ids)
    mask = ids[0] == image_token_id
    assert int(mask.sum()) == image_embeds.shape[0], \
        f"image token {int(mask.sum())} != embed {image_embeds.shape[0]}"
    emb[0, mask] = image_embeds.to(emb.dtype)
    return emb


def build_mm_pos4(ids, image_token_id, grid_thw, merge=2):
    """mrope 4D 位置 (4,1,L) — 语义对齐 Qwen3_5Model.get_rope_index (单图/无 padding)
    dim0=物理 arange; dim1-3: 文本段三维同值递增, 图像段 t/h/w 三维。
    返回 (pos4, next_rope_pos): 多模态 prefill 后 decode 的 rope 逻辑位置起点
    (图像 576 token 只推进 max(H,W)/merge=24 位, 故 ≠ 物理长度 L)"""
    dev = ids.device
    L = ids.shape[1]
    img_mask = ids[0] == image_token_id
    assert img_mask.any(), "无 image token"
    t = int(grid_thw[0, 0])
    h = int(grid_thw[0, 1]) // merge
    w = int(grid_thw[0, 2]) // merge
    thw = torch.zeros(3, 1, L, dtype=torch.long, device=dev)
    segs, i = [], 0
    while i < L:                                         # 按 图像/文本 分段
        j = i
        while j < L and bool(img_mask[j]) == bool(img_mask[i]):
            j += 1
        segs.append((bool(img_mask[i]), i, j))
        i = j
    cur = 0
    for is_img, s, e in segs:
        n = e - s
        if not is_img:
            thw[:, :, s:e] = torch.arange(n, device=dev).view(1, 1, -1) + cur
            cur += n
        else:
            assert n == t * h * w, f"image 段长 {n} != grid {t}x{h}x{w}"
            ht = torch.arange(t, device=dev)
            hh = torch.arange(h, device=dev) + cur
            hw = torch.arange(w, device=dev) + cur
            g = torch.stack(torch.meshgrid(ht, hh, hw, indexing="ij"),
                            dim=0).reshape(3, -1)        # (3, t·h·w) w 最快, 对齐参考
            g[0] += cur                                  # t 偏移在 stack 后 (对齐参考)
            thw[:, :, s:e] = g.view(3, 1, n)
            cur += max(h, w)
    pos4 = torch.cat([torch.arange(L, device=dev).view(1, 1, L), thw], dim=0)
    return pos4, cur


# ---------------- R1+R2 组合快速路径 (2026-08-17 定稿) ----------------
# R1: StaticCache + CUDA Graph 捕获整步 decode —— 压 CPU dispatch (3.21×)
# R2: fused DeltaNet decode kernel —— 压 48 层线性注意力 dtype churn (1.78×)
# 组合实测 21.78 tok/s (baseline 5.78, 3.77×), 63/63 token 一致


def build_model():
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from accelerate import init_empty_weights

    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cfg = AutoConfig.from_pretrained(MODEL_DIR)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(cfg.text_config, dtype=torch.float16)
    index = json.load(open(IDX, encoding="utf-8"))
    misc = np.load(MISC)
    emb_mode = os.environ.get(
        "VF_EMBED", "nvfp4" if os.path.exists(EMB_P) and os.path.exists(EMB_S)
        else "fp16")
    if emb_mode == "nvfp4":
        embed = PackedEmb(EMB_P, EMB_S)
        print(f"  embed: NVFP4 packed {embed.packed.nbytes>>20}MB+"
              f"{embed.scales.nbytes>>20}MB (RAM 省 1.69GB)", flush=True)
    else:
        embed = torch.from_numpy(np.asarray(np.load(EMB, mmap_mode="r")))
        print("  embed: fp16 (VF_EMBED=fp16)", flush=True)
    model.model.embed_tokens = QEmbedding(embed)
    n_patch = patch_linears(model, index, misc)
    model.to_empty(device="cuda")
    n_fill, skipped = fill_rest(model, misc, embed if emb_mode == "fp16" else None)
    n_rope = rebuild_rope(model)
    model.eval()
    torch.cuda.synchronize()
    vram = torch.cuda.memory_allocated() / 1e9
    free, _total = torch.cuda.mem_get_info()
    print(f"  patch {n_patch} Linear | fill {n_fill} param | rope {n_rope} | "
          f"torch VRAM {vram:.2f}GB + dll {UPLOAD_BYTES[0]/1e9:.2f}GB | "
          f"剩余 {free/1e9:.2f}GB", flush=True)
    if skipped:
        print("  skipped:", skipped[:10], flush=True)
    return tok, model


def patch_fused_deltanet(model, cache, mtp=True):
    """R2: 48 层 linear_attention 换 fused decode kernel。
    必须在 prefill 之后调用 —— 从 cache 抓 conv/recurrent 初始状态到 _vf 自管 buffer
    mtp=False: 不分配 _vf_S_bak/_vf_conv_bak (48层省 151MiB, 16G 卡 server 专用;
    纯 T=1 graph decode 无回滚需求, T=2 分支不可用 — 仅 mtp=True 的 CLI/调试路径可走)"""
    dn = ctypes.CDLL(os.path.join(KDIR, "nvfp4_dn_fused.dll"))
    dn.launch_dn_fused.argtypes = [ctypes.c_void_p] * 12 + [ctypes.c_float, ctypes.c_void_p]
    dn.launch_dn_fused.restype = ctypes.c_int
    mods = []
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "linear_attention":
            continue
        mod = lyr.linear_attn
        lay = cache.layers[i]
        conv = lay.conv_states[0].reshape(-1)
        rec = lay.recurrent_states[0]                     # (1,48,128,128) k×v
        mod._vf_conv = torch.zeros(10240 * 4, dtype=torch.float16, device="cuda")
        mod._vf_conv.copy_(conv.to(torch.float16))        # 布局 c*4+t (4 帧滚动)
        mod._vf_S = rec.transpose(-1, -2).contiguous().float().reshape(-1).clone()
        mod._vf_scr = torch.zeros(10240, dtype=torch.float32, device="cuda")
        mod._vf_out = torch.zeros(6144, dtype=torch.float16, device="cuda")
        mod._vf_out2 = torch.zeros(6144, dtype=torch.float16, device="cuda")
        mod._vf_S_bak = torch.zeros_like(mod._vf_S) if mtp else None
        mod._vf_conv_bak = torch.zeros_like(mod._vf_conv) if mtp else None
        mod._vf_eps = float(getattr(mod.norm, "variance_epsilon", 1e-6))
        if not getattr(mod, "_vf_fast", False):          # 二次 patch (换 cache) 防包裹自身
            mod._vf_orig = mod.forward
        mod._vf_fast = True
        mods.append(mod)

    def fast_fwd(mod):
        def fwd(hidden_states, cache_params=None, attention_mask=None, **kw):
            Tn = hidden_states.shape[1]
            if Tn > 2:                                    # prefill/长序列走原路径
                return mod._vf_orig(hidden_states, cache_params=cache_params,
                                    attention_mask=attention_mask, **kw)
            st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            p = ctypes.c_void_p
            if Tn == 1:                  # T=1 常规 decode: 调用前备份 (拒绝回滚)
                if mod._vf_S_bak is not None:   # mtp=False: 无备份, graph 内零冗余拷贝
                    mod._vf_S_bak.copy_(mod._vf_S)
                    mod._vf_conv_bak.copy_(mod._vf_conv)
                mixed = mod.in_proj_qkv(hidden_states)     # (1,1,M)
                z = mod.in_proj_z(hidden_states)
                b = mod.in_proj_b(hidden_states)
                a = mod.in_proj_a(hidden_states)
                rc = dn.launch_dn_fused(
                    p(mixed.data_ptr()), p(z.data_ptr()), p(b.data_ptr()), p(a.data_ptr()),
                    p(mod.conv1d.weight.data_ptr()), p(mod._vf_conv.data_ptr()),
                    p(mod.A_log.data_ptr()), p(mod.dt_bias.data_ptr()),
                    p(mod.norm.weight.data_ptr()), p(mod._vf_S.data_ptr()),
                    p(mod._vf_scr.data_ptr()), p(mod._vf_out.data_ptr()),
                    mod._vf_eps, st)
                assert rc == 0, f"dn_fused rc={rc}"
                return mod.out_proj(mod._vf_out.view(1, 1, -1))
            # T=2 MTP 链式 verify: 投影一次走 GEMV2 (4 次权重读 vs 8 次 GEMV),
            # dn_fused 仍逐 token (递归状态); token0 后快照 (拒绝回滚)
            mixed = mod.in_proj_qkv(hidden_states)         # (1,2,M)
            z = mod.in_proj_z(hidden_states)
            b = mod.in_proj_b(hidden_states)
            a = mod.in_proj_a(hidden_states)
            outs = []
            for t in range(2):
                obuf = mod._vf_out if t == 0 else mod._vf_out2
                rc = dn.launch_dn_fused(
                    p(mixed[:, t].data_ptr()), p(z[:, t].data_ptr()),
                    p(b[:, t].data_ptr()), p(a[:, t].data_ptr()),
                    p(mod.conv1d.weight.data_ptr()), p(mod._vf_conv.data_ptr()),
                    p(mod.A_log.data_ptr()), p(mod.dt_bias.data_ptr()),
                    p(mod.norm.weight.data_ptr()), p(mod._vf_S.data_ptr()),
                    p(mod._vf_scr.data_ptr()), p(obuf.data_ptr()),
                    mod._vf_eps, st)
                assert rc == 0, f"dn_fused rc={rc}"
                if t == 0 and mod._vf_S_bak is not None:
                    # None 守卫同 T=1 分支: mtp=False 无备份缓冲 —— MMLU 跑分
                    # 实证 prompt len % 64 == 2 时 prefill 尾 chunk T=2 误入
                    # 本分支, copy_ None 崩 500 (5/200 题, 确定性复现)
                    mod._vf_S_bak.copy_(mod._vf_S)
                    mod._vf_conv_bak.copy_(mod._vf_conv)
                outs.append(obuf.view(1, 1, -1))
            return mod.out_proj(torch.cat(outs, dim=1))
        return fwd

    for mod in mods:
        mod.forward = fast_fwd(mod)
    return mods


# KV 快照暂存缓冲: snap/restore 的 GPU slice 拷贝若直连 CPU, PyTorch 会物化
# 24MB/层 连续临时 (17K ctx 实测 OOM 点, nvfp4-021)。启动时预分配一次, 高峰零分配。
_KV_STAGE = {"buf": None, "elems": 0}


def ensure_kv_stage(model, max_len):
    """模型就绪后 (显存余量充足时) 预分配 KV 前缀暂存缓冲, 单层 K 或 V 复用"""
    cfg = model.config
    H = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    D = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n = H * max_len * D
    if _KV_STAGE["buf"] is None or _KV_STAGE["elems"] < n:
        _KV_STAGE["buf"] = torch.empty(n, dtype=torch.float16, device="cuda")
        _KV_STAGE["elems"] = n
    return _KV_STAGE["buf"]


def snap_decode_state(model, cache, prompt_len):
    """graph warmup 前快照 attention KV 前缀 + cumulative_length (发现39: 时序关键)
    快照驻 CPU: 8K 上下文时 KV 前缀 470MB, GPU clone 会吃爆余量 (nvfp4-012 教训);
    有暂存缓冲走 GPU slice→连续 stage→CPU (避免物化 24MB/层临时), 无则退回直拷"""
    snap = []
    buf = _KV_STAGE["buf"]
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type == "linear_attention":
            continue
        lay = cache.layers[i]
        k_seg, v_seg = lay.keys[:, :, :prompt_len], lay.values[:, :, :prompt_len]
        if buf is not None and buf.numel() >= max(k_seg.numel(), v_seg.numel()):
            st = buf[:k_seg.numel()].view_as(k_seg)
            st.copy_(k_seg)                          # GPU→GPU, 目标连续, 零分配
            k_cpu = st.cpu()
            st = buf[:v_seg.numel()].view_as(v_seg)
            st.copy_(v_seg)
            v_cpu = st.cpu()
        else:                                        # CLI 单发路径: 直拷 (物化临时)
            k_cpu = k_seg.clone().cpu()
            v_cpu = v_seg.clone().cpu()
        snap.append((i, k_cpu, v_cpu, lay.cumulative_length.clone().cpu()))
    return snap


def restore_decode_state(model, cache, snap, prompt_len):
    buf = _KV_STAGE["buf"]
    for i, k, v, c in snap:
        lay = cache.layers[i]
        if buf is not None and buf.numel() >= max(k.numel(), v.numel()):
            st = buf[:k.numel()].view_as(k)
            st.copy_(k)                              # CPU→GPU 双连续, 纯 H2D
            lay.keys[:, :, :prompt_len].copy_(st)    # GPU→GPU copy kernel, 零分配
            st = buf[:v.numel()].view_as(v)
            st.copy_(v)
            lay.values[:, :, :prompt_len].copy_(st)
        else:
            lay.keys[:, :, :prompt_len].copy_(k)
            lay.values[:, :, :prompt_len].copy_(v)
        lay.cumulative_length.copy_(c)


def refresh_deltanet_state(model, cache):
    """全局 cache 架构: 每请求 prefill 后把 cache 的 conv/recurrent 状态刷进
    _vf 缓冲 (copy_ 复用既有 buffer, 零新分配)。必须在 patch_fused_deltanet 之后。
    strided 源 + dtype cast 由 copy_ 单 kernel 完成, 不物化中间张量"""
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "linear_attention":
            continue
        mod = lyr.linear_attn
        lay = cache.layers[i]
        rec = lay.recurrent_states[0]                # (1,48,128,128) k×v
        rt = rec.transpose(-1, -2)                   # view, 无物化
        mod._vf_S.view(rt.shape).copy_(rt)           # fp16→fp32 cast 单 kernel
        mod._vf_conv.copy_(lay.conv_states[0].reshape(-1))


def patch_attn_decode(model, cache, max_len=3072):
    """P5-B: 16 层 full_attention 的 T=1 前向换自研 GQA decode kernel
    (nvfp4_attn_decode.dll, sm_70)。语义: score=q·k*scale → softmax fp32 →
    Σp·v; L=update 后 cumulative_length (graph replay 动态)。
    T>1 (prefill/MTP) 走原 forward。NCU: fmha_cutlassF 5010μs/步 → <150μs/步。
    依据 DLL shared 尺寸绑定, max_len 必须为 3072。"""
    lib = ctypes.CDLL(os.path.join(KDIR, "nvfp4_attn_decode.dll"))
    lib.launch_attn_decode.argtypes = [ctypes.c_void_p] * 6 + \
        [ctypes.c_int, ctypes.c_float, ctypes.c_void_p]
    lib.launch_attn_decode.restype = ctypes.c_int
    p = ctypes.c_void_p
    from transformers.models.qwen3_5.modeling_qwen3_5 import apply_rotary_pos_emb
    mods = []
    for i, lyr in enumerate(model.model.layers):
        if lyr.block_type != "full_attention":
            continue
        mod, lay = lyr.self_attn, cache.layers[i]
        mod._vf_attn_out = torch.zeros(mod.config.num_attention_heads * mod.head_dim,
                                       dtype=torch.float16, device="cuda")
        if not getattr(mod, "_vf_attn", False):      # 二次 patch 防包裹自身
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
            input_shape = hidden_states.shape[:-1]          # (1,1)
            hidden_shape = (*input_shape, -1, D)            # (1,1,-1,256)
            qg = mod.q_proj(hidden_states).view(*input_shape, -1, D * 2)
            query_states, gate = torch.chunk(qg, 2, dim=-1)
            gate = gate.reshape(*input_shape, -1)           # (1,1,6144)
            query_states = mod.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
            key_states = mod.k_norm(mod.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
            value_states = mod.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(
                query_states, key_states, cos, sin)
            # transpose 仅逻辑 shape, 内存仍是每头连续 256 → q_ptr[h*256+d]
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
    return mods


class GraphDecoder:
    """R1: 整步 decode 的 CUDA Graph —— forward + argmax + logits + pos 递增全捕获。
    QEmbedding 的 CPU gather 留 graph 外 (pinned → H2D → replay)
    P0: use_pos4=True 时显式传 mrope position_ids (4,1,1) — 文本路径与内部
    arange+past_seen 数值等价; 多模态 prefill 后 rope 逻辑位置 ≠ 物理槽位
    (图像 576 token 只推进 24 位), 用 rope_pos 跟踪逻辑位置"""

    def __init__(self, model, cache, first_pos, hid, n_warmup=3,
                 static_pos=None, use_pos4=True):
        self.first_pos = first_pos
        self.rope_pos = first_pos                           # decode 的 rope 逻辑位置
        self.pos4 = torch.zeros(4, 1, 1, dtype=torch.long, device="cuda")
        self.static_embeds = torch.zeros(1, 1, hid, dtype=torch.float16, device="cuda")
        # static_pos 可外部共享 (MTP: 与 GraphDecoder2 同一 (2,) tensor, 取 [:1] view)
        if static_pos is None:
            self.static_pos = torch.tensor([first_pos, first_pos + 1], device="cuda")
            self._own_pos = True
        else:
            self.static_pos = static_pos                      # (2,) 共享
            self._own_pos = False
        self.pos2 = self.static_pos[:1].view(1)               # cache_position (1,)
        V = model.lm_head.M
        self.static_logits = torch.zeros(1, V, dtype=torch.float16, device="cuda")
        self.static_next = torch.zeros(1, dtype=torch.long, device="cuda")
        self.pin = torch.empty(hid, dtype=torch.float16, pin_memory=True)

        def one_step():
            kw = {"position_ids": self.pos4} if use_pos4 else {}
            o = model(inputs_embeds=self.static_embeds, past_key_values=cache,
                      cache_position=self.pos2, use_cache=True, **kw)
            self.static_logits.copy_(o.logits[:, -1, :])
            self.static_next.copy_(o.logits[:, -1, :].argmax(dim=-1))
            self.static_pos.add_(1)     # 原地: += 会重绑定闭包变量 (发现39)

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(n_warmup):   # warmup 会污染状态, 由调用方恢复快照
                one_step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        # warmup 临时缓冲归还缓存池后释放回驱动: graph 私有池需要整块物理显存,
        # 16GB 卡余量 <120MB 时碎片必炸 (nvfp4-021 实测 capture_end OOM)
        torch.cuda.empty_cache()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            one_step()

    def reset_pos(self):
        self.static_pos.fill_(self.first_pos)
        self.rope_pos = self.first_pos

    def set_pos(self, p):
        """全局 graph 复用: 每请求把 decode 起点拨到本请求 prompt 长度 L。
        graph 内只捕获 '对 static_pos/cache_position 寻址' 的算子, 值随 replay 动态"""
        self.first_pos = p
        self.rope_pos = p
        self.static_pos.fill_(p)

    def step(self, tok_id, emb_w):
        self.pin.copy_(emb_w[tok_id])                        # CPU gather (graph 外)
        self.static_embeds.copy_(self.pin.view(1, 1, -1), non_blocking=True)
        self.pos4.fill_(self.rope_pos)                       # graph 外填充 (静态输入)
        self.graph.replay()
        self.rope_pos += 1
        return int(self.static_next.item())

    def step_logits(self, tok_id, emb_w):
        """MTP verify: replay 后返回 (argmax, 完整 logits GPU tensor)"""
        self.pin.copy_(emb_w[tok_id])
        self.static_embeds.copy_(self.pin.view(1, 1, -1), non_blocking=True)
        self.pos4.fill_(self.rope_pos)
        self.graph.replay()
        self.rope_pos += 1
        return int(self.static_next.item()), self.static_logits


class GraphDecoder2:
    """MTP T=2 链式 verify 的 CUDA Graph: forward [tokA, tokB] 一次。
    logits[0]=P(·|..,A) 判 B; logits[1]=P(·|..,A,B) 给 bonus。
    与 GraphDecoder 共享 static_pos (2,) —— 外部每轮校准 [wpos, wpos+1]"""

    def __init__(self, model, cache, first_pos, hid, static_pos, n_warmup=3):
        self.static_embeds = torch.zeros(1, 2, hid, dtype=torch.float16, device="cuda")
        self.static_pos = static_pos                          # (2,) 共享 tensor
        V = model.lm_head.M
        self.static_logits = torch.zeros(2, V, dtype=torch.float16, device="cuda")
        self.static_next = torch.zeros(2, dtype=torch.long, device="cuda")
        self.pinA = torch.empty(hid, dtype=torch.float16, pin_memory=True)
        self.pinB = torch.empty(hid, dtype=torch.float16, pin_memory=True)

        def one_step2():
            o = model(inputs_embeds=self.static_embeds, past_key_values=cache,
                      cache_position=self.static_pos, use_cache=True)
            lg = o.logits[:, -2:, :][0]                       # (2, V)
            self.static_logits.copy_(lg)
            self.static_next.copy_(lg.argmax(dim=-1))
            self.static_pos.add_(1)                           # 每元素 +1

        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(n_warmup):
                one_step2()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph), torch.no_grad():
            one_step2()

    def step2(self, tokA, tokB, emb_w):
        """返回 (argmax(2,), logits(2,V)) — graph 外 CPU gather + H2D"""
        self.pinA.copy_(emb_w[tokA])
        self.pinB.copy_(emb_w[tokB])
        self.static_embeds[:, 0].copy_(self.pinA.view(1, -1), non_blocking=True)
        self.static_embeds[:, 1].copy_(self.pinB.view(1, -1), non_blocking=True)
        self.graph.replay()
        return self.static_next, self.static_logits


def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好，请介绍一下你自己。"
    ngen = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    maxlen = int(sys.argv[3]) if len(sys.argv) > 3 else 1024
    from transformers import StaticCache

    print("[1/5] 建骨架 + NVFP4 权重上传 ...", flush=True)
    t0 = time.time()
    tok, model = build_model()
    print(f"  构建 {time.time()-t0:.0f}s", flush=True)

    print("[2/5] prefill (StaticCache) ...", flush=True)
    text = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True,
                                   enable_thinking=False)
    ids = tok(text, return_tensors="pt").input_ids.cuda()
    L = ids.shape[1]
    assert L + ngen <= maxlen, f"prompt({L})+gen({ngen}) 超 maxlen({maxlen})"
    cache = StaticCache(config=model.config, max_cache_len=maxlen)
    t1 = time.time()
    # 分块 prefill (随读随放): 每块 512, KV 进 cache, 中间激活即放。
    # 修正: 全序列 logits 并非 prefill OOM 主犯 (lm_head 不调用仍 OOM@2050);
    # 真凶 = sdpa math backend 物化 L×L attn 矩阵 + DeltaNet scan 中间量。
    # 分块把激活峰值从 L 压到 chunk, 8K 上下文实测通过 (token 级一致验证)。
    CHUNK = 512
    first = None
    with torch.no_grad():
        for i in range(0, L, CHUNK):
            j = min(i + CHUNK, L)
            out = model(ids[:, i:j], past_key_values=cache, use_cache=True,
                        cache_position=torch.arange(i, j, device="cuda"),
                        logits_to_keep=1)
            if j == L:
                first = int(out.logits[:, -1, :].argmax(dim=-1).item())
    torch.cuda.synchronize()
    dt1 = time.time() - t1
    print(f"  prefill {L} tok in {dt1:.2f}s ({L/dt1:.0f} tok/s)", flush=True)

    print("[3/5] R2 fused DeltaNet patch ...", flush=True)
    mods = patch_fused_deltanet(model, cache)
    # 快照驻 CPU: 8K 时 GPU 余量不足, GPU clone 吃爆 (nvfp4-012 教训)
    vf_snap = [(m._vf_conv.clone().cpu(), m._vf_S.clone().cpu())
               for m in mods]
    snap = snap_decode_state(model, cache, L)
    print(f"  {len(mods)} 层已换 fused kernel", flush=True)

    print("[4/5] R1 CUDA Graph 捕获 ...", flush=True)
    emb_w = model.model.embed_tokens.weight_cpu
    dec = GraphDecoder(model, cache, L, emb_w.shape[1])
    restore_decode_state(model, cache, snap, L)          # 恢复 warmup 污染
    for m, (c, s) in zip(mods, vf_snap):
        m._vf_conv.copy_(c)
        m._vf_S.copy_(s)
    dec.reset_pos()
    torch.cuda.synchronize()

    print(f"[5/5] decode {ngen} tok (R1+R2) ...", flush=True)
    gen = [first]
    cur = first
    eos = tok.eos_token_id
    t2 = time.perf_counter()
    with torch.no_grad():
        for _ in range(ngen):
            cur = dec.step(cur, emb_w)
            gen.append(cur)
            if eos is not None and cur == eos:
                break
    torch.cuda.synchronize()
    dt = time.perf_counter() - t2
    n_out = len(gen) - 1
    print(f"  decode {n_out} tok in {dt:.2f}s -> {n_out/dt:.1f} tok/s (R1+R2)",
          flush=True)
    print("生成内容:", flush=True)
    print(tok.decode(gen), flush=True)


if __name__ == "__main__":
    main()


def patch_rmsnorm(model):
    """P5-A-2: Qwen3_5RMSNorm 全站融合 (161 处/步 × 10 内核 → 1 内核)。
    语义 (modeling_qwen3_5.py L749-754, zero-centered):
      out = f16( f32(x) · rsqrt(mean(x²)+eps) · (1+w_f32) )
    w1 = 1+w 预计算一次; x 允许 chunk view 的行间 gap (stride(-2))。
    NCU: torch 链含每步重算 w.float()/1.0+w, 5.5ms/步 → ~0.5ms。"""
    lib = ctypes.CDLL(os.path.join(KDIR, "nvfp4_rmsnorm.dll"))
    lib.launch_rmsnorm.argtypes = [ctypes.c_void_p, ctypes.c_longlong,
                                   ctypes.c_void_p, ctypes.c_void_p,
                                   ctypes.c_int, ctypes.c_int,
                                   ctypes.c_float, ctypes.c_void_p]
    lib.launch_rmsnorm.restype = ctypes.c_int
    norms = []
    for mod in model.modules():
        if mod.__class__.__name__ != "Qwen3_5RMSNorm":
            continue
        mod._vf_w1 = (1.0 + mod.weight.float()).contiguous()
        mod._vf_out = None
        if not getattr(mod, "_vf_norm", False):   # 二次 patch 防包裹自身
            mod._vf_norm_orig = mod.forward
        mod._vf_norm = True
        norms.append(mod)

    def fast_norm(mod):
        def fwd(x):
            n = x.shape[-1]
            # T>1 (prefill) 走原路径: 避免 (T,5120) 大输出缓冲驻留 130 处 → OOM
            if x.dim() >= 2 and x.shape[1] != 1:
                return mod._vf_norm_orig(x)
            rows = x.numel() // n
            if rows <= 0 or x.stride(-1) != 1:
                return mod._vf_norm_orig(x)
            if mod._vf_out is None or tuple(mod._vf_out.shape) != (rows, n):
                mod._vf_out = torch.empty((rows, n), dtype=x.dtype,
                                          device=x.device)
            st = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
            rc = lib.launch_rmsnorm(
                ctypes.c_void_p(x.data_ptr()), x.stride(-2),
                ctypes.c_void_p(mod._vf_w1.data_ptr()),
                ctypes.c_void_p(mod._vf_out.data_ptr()),
                rows, n, float(mod.eps), st)
            assert rc == 0, f"rmsnorm rc={rc}"
            return mod._vf_out.view(x.shape)
        return fwd

    for mod in norms:
        mod.forward = fast_norm(mod)
    return norms
