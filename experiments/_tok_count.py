# -*- coding: utf-8 -*-
"""量 token 数 (临时)"""
import sys

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "pipeline"))
sys.stdout.reconfigure(encoding="utf-8")
from _qwen38_infer import build_model

tok, _ = build_model()
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "_qwen38_p4a_lru.py"), encoding="utf-8").read()
ns = {}
head = src.split("SUFFIX = ")[0].split("CHUNK = 512")[1]
exec("CHUNK=512" + head, ns)
for n in ("P1_TEXT", "P2_TEXT", "P3_TEXT"):
    t = ns[n]
    print(n, "chars", len(t), "tokens", len(tok(t)["input_ids"]))
