#!/usr/bin/env python3
"""单条延迟：冷（每次新 prompt，避开 radix cache）与热（同一输入重复）各测几次。"""
import base64
import glob
import json
import statistics
import time
import urllib.request
import uuid

BASE = "http://127.0.0.1:19100"
IMG = sorted(glob.glob("/root/hygon/trident/eval/ChartQA/ChartQA/images/*.png"))[0]
B64 = base64.b64encode(open(IMG, "rb").read()).decode()


def post(inp):
    body = {"model": "trident-qwen3vl-2b", "input": inp}
    req = urllib.request.Request(BASE + "/v1/embeddings", data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as r:
        r.read()
    return time.time() - t0


def run(label, mk, n=5):
    ts = []
    for _ in range(n):
        ts.append(post(mk()))
    print(json.dumps({"case": label, "n": n, "median_s": round(statistics.median(ts), 3),
                      "min_s": round(min(ts), 3), "max_s": round(max(ts), 3)}))


run("text_cold", lambda: [{"text": "query " + uuid.uuid4().hex}])
run("text_warm_repeat", lambda: [{"text": "query fixed"}])
run("image_cold", lambda: [{"text": "doc " + uuid.uuid4().hex, "image": "data:image/png;base64," + B64}])
run("image_warm_repeat", lambda: [{"text": "doc fixed", "image": "data:image/png;base64," + B64}])
