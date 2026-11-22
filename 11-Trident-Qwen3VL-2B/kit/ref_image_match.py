#!/usr/bin/env python3
"""图像/图文输入：补一条空的 assistant 轮，让 sglang 也带 assistant 前缀，再和官方比对。

结论用：文本路径加了 assistant 前缀后 cos=0.9999，说明权重/池化/归一化一致；
这里验证图像路径能否用同样手法对齐。
"""
import base64
import glob
import json
import math
import sys
import time
import urllib.request

CKPT = "/root/hygon/models/Trident-Qwen3VL-2B"
BASE = "http://127.0.0.1:19100"
OUT = "/root/hygon/trident/evidence/image_prefix_match.json"
IMG = sorted(glob.glob("/root/hygon/a2wan/data/vbench_i2v/crop/16-9/*.jpg"))[0]


def post(inp):
    body = {"model": "trident-qwen3vl-2b", "input": inp}
    req = urllib.request.Request(BASE + "/v1/embeddings", data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read().decode())
    return d["data"][0]["embedding"], time.time() - t0


def cos(a, b):
    return sum(x * y for x, y in zip(a, b)) / (
        math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b)) + 1e-12)


def main():
    sys.path.insert(0, CKPT)
    from trident_embedder import TridentEmbedder

    m = TridentEmbedder(CKPT)
    instr = m.instructions["doc"]
    b64 = base64.b64encode(open(IMG, "rb").read()).decode()
    img_item = {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}}

    cases = {
        "image_only": {"text": None, "images": [IMG]},
        "image_text": {"text": "Flowing abstract colours filling the frame.", "images": [IMG]},
    }
    refs = m.process(list(cases.values()), role="doc")

    rows = []
    for i, (name, item) in enumerate(cases.items()):
        content = [img_item]
        if item["text"]:
            content.append({"type": "text", "text": item["text"]})
        base = [{"role": "system", "content": [{"type": "text", "text": instr}]},
                {"role": "user", "content": content}]
        for tag, extra in (("no_assistant", []),
                           ("with_assistant", [{"role": "assistant",
                                                "content": [{"type": "text", "text": ""}]}])):
            try:
                vec, dt = post(base + extra)
                rows.append({"case": name, "variant": tag, "latency_s": round(dt, 3),
                             "cos_vs_ref": round(cos(vec, refs[i].tolist()), 6)})
            except Exception as e:  # noqa: BLE001
                rows.append({"case": name, "variant": tag, "error": repr(e)[:200]})
            print(rows[-1])

    json.dump({"image": IMG, "rows": rows}, open(OUT, "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("-> " + OUT)


if __name__ == "__main__":
    main()
