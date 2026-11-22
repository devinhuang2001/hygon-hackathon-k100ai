#!/usr/bin/env python3
"""用 sglang 官方的 MultimodalEmbeddingInput 形态（{"text","image"}）验证图像是否生效。"""
import base64
import glob
import json
import math
import urllib.request

BASE = "http://127.0.0.1:19100"
MODEL = "trident-qwen3vl-2b"
INSTR = "Represent the candidate content for retrieval."
IMGS = sorted(glob.glob("/root/hygon/trident/eval/ChartQA/ChartQA/images/*.png"))


def post(inp):
    body = {"model": MODEL, "input": inp}
    req = urllib.request.Request(BASE + "/v1/embeddings", data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read().decode())["data"][0]["embedding"]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)


def uri(p):
    return "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()


v_text = post([{"text": INSTR}])
v_img0 = post([{"text": INSTR, "image": uri(IMGS[0])}])
v_img1 = post([{"text": INSTR, "image": uri(IMGS[1])}])
v_both = post([{"text": INSTR, "image": uri(IMGS[0])}])
print(json.dumps({
    "cos_text_vs_img0": round(cos(v_text, v_img0), 6),
    "cos_img0_vs_img1": round(cos(v_img0, v_img1), 6),
    "repeat_same_input": round(cos(v_img0, v_both), 6),
    "dim": len(v_text),
}, ensure_ascii=False))
