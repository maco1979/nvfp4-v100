"""v3 边界尺寸崩溃最小复现: 100x100x112 (compute-sanitizer 用)"""
import ctypes, os, sys
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "codec"))
from nvfp4_codec import encode_fp16_to_nvfp4, decode_nvfp4_to_fp16

BASE = os.path.join(_ROOT, "kernels")
lib = ctypes.CDLL(os.path.join(BASE, "nvfp4_cuda_v3.dll"))
lib.gpu_malloc.argtypes = [ctypes.c_size_t]
lib.gpu_malloc.restype = ctypes.c_void_p
lib.gpu_free.argtypes = [ctypes.c_void_p]
lib.gpu_memcpy_h2d.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_h2d.restype = ctypes.c_int
lib.gpu_memcpy_d2h.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
lib.gpu_memcpy_d2h.restype = ctypes.c_int
lib.launch_nvfp4_gemmtc_v3.argtypes = [
    ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int]
lib.launch_nvfp4_gemmtc_v3.restype = ctypes.c_int

M, N, K = (int(x) for x in (sys.argv[1:4] if len(sys.argv) > 3 else (100, 100, 112)))
rng = np.random.default_rng(42)
w = (rng.standard_normal((M, K), dtype=np.float32) * 2.0).astype(np.float16)
a = (rng.standard_normal((K, N), dtype=np.float32) * 0.5).astype(np.float16)
packed, scales = encode_fp16_to_nvfp4(w)

pb, sb, ab = (packed.astype(np.uint8).tobytes(), scales.astype(np.uint8).tobytes(),
              a.tobytes())
d_p, d_s, d_a = lib.gpu_malloc(len(pb)), lib.gpu_malloc(len(sb)), lib.gpu_malloc(len(ab))
d_o = lib.gpu_malloc(M * N * 2)
lib.gpu_memcpy_h2d(d_p, pb, len(pb))
lib.gpu_memcpy_h2d(d_s, sb, len(sb))
lib.gpu_memcpy_h2d(d_a, ab, len(ab))
ret = lib.launch_nvfp4_gemmtc_v3(d_p, d_s, d_a, d_o, M, N, K)
print(f"kernel ret={ret}")
out = np.empty(M * N, dtype=np.float16)
lib.gpu_memcpy_d2h(out.ctypes.data, d_o, M * N * 2)
out = out.reshape(M, N)

w_ref = decode_nvfp4_to_fp16(packed, scales).astype(np.float32)
ref = w_ref @ a.astype(np.float32)
err = np.abs(out.astype(np.float32) - ref)
print(f"max_abs={err.max():.4f} mean_rel={(err/(np.abs(ref)+1e-3)).mean():.4f}")
