#!/usr/bin/env python3
"""检查 sglang 嵌入模式是否真的把图像编码进去了。

比较：带图请求的向量 vs 同一 prompt 但无图的向量。
余弦 ≈ 1 → 图像被忽略（只是文本指令的向量）。
"""
import base64
import glob
import json
import math
import urllib.request

BASE = "http://127.0.0.1:19100"
MODEL = "trident-qwen3vl-2b"
INSTR = "Represent the candidate content for retrieval."
IMG = sorted(glob.glob("/root/hygon/trident/eval/ChartQA/ChartQA/images/*.png"))[0]


def post(inp):
    body = {"model": MODEL, "input": inp}
    req = urllib.request.Request(BASE + "/v1/embeddings", data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())["data"][0]["embedding"]


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)


b64 = base64.b64encode(open(IMG, "rb").read()).decode()
sys_only = [{"role": "system", "content": [{"type": "text", "text": INSTR}]},
            {"role": "user", "content": [{"type": "text", "text": "content"}]}]
with_img = [{"role": "system", "content": [{"type": "text", "text": INSTR}]},
            {"role": "user", "content": [{"type": "image_url",
                                          "image_url": {"url": "data:image/png;base64," + b64}}]}]
a = post(sys_only)
b = post(with_img)
print(json.dumps({"image": IMG, "cos_text_only_vs_with_image": round(cos(a, b), 6),
                  "dim": len(a)}))

# 两张不同图的向量互相比较，看是否也一样
img2 = sorted(glob.glob("/root/hygon/trident/eval/ChartQA/ChartQA/images/*.png"))[7]
b642 = base64.b64encode(open(img2, "rb").read()).decode()
c = post([{"role": "system", "content": [{"type": "text", "text": INSTR}]},
          {"role": "user", "content": [{"type": "image_url",
                                        "image_url": {"url": "data:image/png;base64," + b642}}]}])
print(json.dumps({"cos_two_different_images": round(cos(b, c), 6)}))
