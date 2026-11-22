#!/usr/bin/env python3
"""Trident-Qwen3VL-2B on SGLang 的嵌入接口冒烟：文本 / 图像 / 图文混合。

用法：python3 smoke.py [base_url]
输出：evidence/smoke.jsonl（每行一个请求的结果与向量摘要）
"""
import base64
import glob
import json
import math
import os
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:19100"
MODEL = "trident-qwen3vl-2b"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evidence", "smoke.jsonl")


def post(path, body, timeout=180):
    req = urllib.request.Request(BASE + path, data=json.dumps(body).encode(), method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
        return data, time.time() - t0, None
    except urllib.error.HTTPError as e:
        return None, time.time() - t0, "HTTP %s: %s" % (e.code, e.read().decode()[:400])
    except Exception as e:  # noqa: BLE001
        return None, time.time() - t0, repr(e)


def summarize(vec):
    norm = math.sqrt(sum(v * v for v in vec))
    return {"dim": len(vec), "l2": round(norm, 6),
            "head": [round(v, 5) for v in vec[:6]]}


def b64_image():
    cands = sorted(glob.glob("/root/hygon/a2wan/data/vbench_i2v/crop/16-9/*.jpg"))
    if not cands:
        return None, None
    p = cands[0]
    return p, base64.b64encode(open(p, "rb").read()).decode()


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fh = open(OUT, "w", encoding="utf-8")

    def rec(name, body, data, dt, err):
        row = {"test": name, "latency_s": round(dt, 3), "error": err}
        if data and "data" in data:
            embs = [d.get("embedding") for d in data["data"]]
            if embs and isinstance(embs[0], list):
                row["n"] = len(embs)
                row["vectors"] = [summarize(v) for v in embs[:2]]
            else:
                row["raw_head"] = str(data)[:200]
        elif data:
            row["raw_head"] = str(data)[:300]
        fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        print(json.dumps(row, ensure_ascii=False))

    # 1) 纯文本 query
    body = {"model": MODEL, "input": ["How many stores did Saint Laurent operate in Western Europe in 2020?"]}
    d, dt, e = post("/v1/embeddings", body)
    rec("text_only", body, d, dt, e)

    # 2) 单图 document（data URI）
    path, b64 = b64_image()
    if b64:
        body = {"model": MODEL, "input": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
            ],
        }]}
        d, dt, e = post("/v1/embeddings", body)
        rec("image_only", {"image": path}, d, dt, e)

        # 3) 图文混合 document
        body = {"model": MODEL, "input": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}},
                {"type": "text", "text": "a photo captured from a video, described for retrieval"},
            ],
        }]}
        d, dt, e = post("/v1/embeddings", body)
        rec("image_text_fused", {"image": path}, d, dt, e)
    fh.close()
    print("-> " + OUT)


if __name__ == "__main__":
    main()
