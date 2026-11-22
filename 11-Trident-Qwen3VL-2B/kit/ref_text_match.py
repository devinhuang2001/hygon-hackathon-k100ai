#!/usr/bin/env python3
"""定位 sglang 与官方 TridentEmbedder 的差异来源（纯文本路径）。

官方 prompt 配置：system 放 instruction、add_generation_prompt=True。
sglang 走 messages 时不一定带 assistant 前缀。于是把官方模板渲染成**字符串**再喂给
sglang 的 /v1/embeddings，逐个变体比对余弦，定位差异是不是只来自 chat template。

用法：python3 ref_text_match.py
"""
import json
import math
import os
import sys
import time
import urllib.request

CKPT = "/root/hygon/models/Trident-Qwen3VL-2B"
BASE = "http://127.0.0.1:19100"
OUT = "/root/hygon/trident/evidence/text_template_match.json"
TEXT = "abstract fluid texture flowing in warm colours"


def post_string(text):
    body = {"model": "trident-qwen3vl-2b", "input": [text]}
    req = urllib.request.Request(BASE + "/v1/embeddings", data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read().decode())
    return d["data"][0]["embedding"], time.time() - t0


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)


def main():
    sys.path.insert(0, CKPT)
    from trident_embedder import TridentEmbedder

    m = TridentEmbedder(CKPT)
    instr = m.instructions["query"]
    ref = m.process([{"text": TEXT, "images": []}], role="query")[0].tolist()
    tok = m.tokenizer

    variants = {}
    conv_sys = [{"role": "system", "content": [{"type": "text", "text": instr}]},
                {"role": "user", "content": [{"type": "text", "text": TEXT}]}]
    conv_nosys = [{"role": "user", "content": [{"type": "text", "text": TEXT}]}]
    variants["sys+gen"] = tok.apply_chat_template(conv_sys, tokenize=False, add_generation_prompt=True)
    variants["sys"] = tok.apply_chat_template(conv_sys, tokenize=False, add_generation_prompt=False)
    variants["nosys+gen"] = tok.apply_chat_template(conv_nosys, tokenize=False, add_generation_prompt=True)
    variants["raw"] = TEXT

    rows = []
    for name, text in variants.items():
        try:
            vec, dt = post_string(text)
            rows.append({"variant": name, "chars": len(text), "latency_s": round(dt, 3),
                         "cos_vs_ref": round(cos(vec, ref), 6),
                         "prompt_head": text[:60].replace("\n", " ")})
        except Exception as e:  # noqa: BLE001
            rows.append({"variant": name, "error": repr(e)[:200]})
        print(rows[-1])

    json.dump({"instruction": instr, "rows": rows}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("-> " + OUT)


if __name__ == "__main__":
    main()
