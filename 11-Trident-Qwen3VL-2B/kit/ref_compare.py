#!/usr/bin/env python3
"""官方 TridentEmbedder（transformers 路径）与 SGLang 嵌入接口的一致性对比。

同一批输入分别走：
  A. 官方 `TridentEmbedder.process()`（system=instruction + user=text/images，bf16）
  B. sglang /v1/embeddings（messages 形式，逐字相同的 instruction）
对比逐条余弦相似度与检索排序。

用法：python3 ref_compare.py      （用 /root/img_root/usr/local/bin/python3.10）
输出：evidence/ref_vs_sglang.json
"""
import base64
import glob
import json
import math
import os
import sys
import time
import urllib.request

CKPT = "/root/hygon/models/Trident-Qwen3VL-2B"
BASE = "http://127.0.0.1:19100"
MODEL = "trident-qwen3vl-2b"
OUT = "/root/hygon/trident/evidence/ref_vs_sglang.json"

QUERY = {"text": "abstract fluid texture flowing in warm colours", "images": []}
IMG = sorted(glob.glob("/root/hygon/a2wan/data/vbench_i2v/crop/16-9/*.jpg"))[0]
DOCS = [
    {"text": "A chart of retail store counts by region and year.", "images": []},
    {"text": None, "images": [IMG]},
    {"text": "Flowing abstract colours filling the frame.", "images": [IMG]},
]


def post(body, timeout=180):
    req = urllib.request.Request(BASE + "/v1/embeddings",
                                data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read().decode())
        return [x["embedding"] for x in d["data"]], time.time() - t0, None
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, repr(e)[:400]


def cos(a, b):
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a))
    db = math.sqrt(sum(y * y for y in b))
    return num / (da * db + 1e-12)


def main():
    sys.path.insert(0, CKPT)
    from trident_embedder import TridentEmbedder  # noqa: E402

    t0 = time.time()
    m = TridentEmbedder(CKPT)
    instr = {k: v for k, v in m.instructions.items()}
    print("[ref] loaded in %.1fs, instructions=%s" % (time.time() - t0, json.dumps(instr)[:200]))

    q_ref = m.process([QUERY], role="query")[0].tolist()
    d_ref = [v.tolist() for v in m.process(DOCS, role="doc")]
    scores_ref = [cos(q_ref, d) for d in d_ref]
    rank_ref = sorted(range(len(d_ref)), key=lambda i: -scores_ref[i])

    # sglang：messages 形式，system 放同一条 instruction
    results = {"query": {}, "docs": [], "ref_scores": scores_ref, "ref_rank": rank_ref}
    for name, role, items in (("query", "query", [QUERY]), ("doc", "doc", DOCS)):
        for i, it in enumerate(items):
            content = []
            if it.get("images"):
                for p in it["images"]:
                    b64 = base64.b64encode(open(p, "rb").read()).decode()
                    content.append({"type": "image_url",
                                    "image_url": {"url": "data:image/jpeg;base64," + b64}})
            if it.get("text"):
                content.append({"type": "text", "text": it["text"]})
            body = {"model": MODEL, "input": [
                {"role": "system", "content": [{"type": "text", "text": instr[role]}]},
                {"role": "user", "content": content},
            ]}
            vecs, dt, err = post(body)
            ref = q_ref if role == "query" else d_ref[i]
            row = {"item": name if role == "query" else "doc%d" % i,
                   "latency_s": round(dt, 3), "error": err}
            if vecs:
                row["cos_vs_ref"] = round(cos(vecs[0], ref), 6)
                row["dim"] = len(vecs[0])
            if role == "query":
                results["query"] = row
            else:
                results["docs"].append(row)

    s_scores = []
    docs = [r for r in results["docs"] if "cos_vs_ref" in r]
    results["note"] = ("cos_vs_ref 是 sglang 向量与官方 embedder 向量的余弦；"
                       "接近 1 表示同一份权重在同一输入上给出一致表示")
    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False, indent=1)[:1500])
    print("-> " + OUT)


if __name__ == "__main__":
    main()
