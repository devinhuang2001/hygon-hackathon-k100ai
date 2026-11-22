#!/usr/bin/env python3
"""诊断 Trident 权重目录的架构声明，以及 sglang 里 VL 嵌入/池化的可用开关。"""
import json
import os
import re

M = "/root/hygon/models/Trident-Qwen3VL-2B"
cfg = json.load(open(os.path.join(M, "config.json"), encoding="utf-8"))
print("keys:", sorted(cfg.keys())[:20])
for k in ("architectures", "auto_map", "model_type", "text_config", "vision_config"):
    v = cfg.get(k)
    if isinstance(v, dict):
        print(k, {kk: v[kk] for kk in list(v)[:6]})
    else:
        print(k, v)
print("files:", sorted(os.listdir(M)))
if os.path.exists(os.path.join(M, "trident_embedder.py")):
    src = open(os.path.join(M, "trident_embedder.py"), encoding="utf-8").read()
    print("classes in trident_embedder:", re.findall(r"^class (\w+)", src, re.M))

D = "/root/hygon/bundle/sglang/rootfs/usr/local/lib/python3.10/dist-packages/sglang"
sa = open(os.path.join(D, "srt/server_args.py"), encoding="utf-8").read()
for pat in ("pooling", "is_embedding", "model_impl"):
    hits = [l.strip() for l in sa.splitlines() if pat in l][:6]
    print("---", pat, "---")
    for h in hits:
        print("  ", h[:150])

for f in ("srt/models/qwen3_vl.py",):
    p = os.path.join(D, f)
    if os.path.exists(p):
        src = open(p, encoding="utf-8").read()
        print("---", f, "pool refs:", [l.strip()[:90] for l in src.splitlines() if "pool" in l.lower()][:8])
