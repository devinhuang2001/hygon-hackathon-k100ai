#!/usr/bin/env python3
"""官方 Trident-Eval（ChartQA, Origin 设定）NDCG@10 评测。

两种后端，同一份数据与同一套指令/模板：
  --backend ref     : 官方 TridentEmbedder（transformers，GPU 参考）
  --backend sglang  : 本件交付的 sglang /v1/embeddings 服务

指标：NDCG@10（每 query 一个相关文档，IDCG=1）、Recall@{1,5,10}。
用法：python3 eval_ndcg.py --backend sglang --workers 8 [--max-queries 300]
"""
import argparse
import base64
import json
import math
import os
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CKPT = "/root/hygon/models/Trident-Qwen3VL-2B"
DATA = "/root/hygon/trident/eval/ChartQA/ChartQA"
BASE = "http://127.0.0.1:19100"
OUT_DIR = "/root/hygon/trident/evidence"


def load(name):
    with open(os.path.join(DATA, name), encoding="utf-8") as fh:
        return [json.loads(l) for l in fh if l.strip()]


def post_emb(body, timeout=180):
    req = urllib.request.Request(BASE + "/v1/embeddings", data=json.dumps(body).encode(),
                                 method="POST")
    req.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.loads(r.read().decode())
    return d["data"][0]["embedding"]


def sglang_text(vec_prompt):
    return post_emb({"model": "trident-qwen3vl-2b", "input": [vec_prompt]})


def sglang_image(path, instr):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    # 必须用 sglang 的 MultimodalEmbeddingInput 形态：{"text","image"}。
    # 用 OpenAI messages（role/content + image_url）会被静默当成纯文本，图像不生效。
    return post_emb({"model": "trident-qwen3vl-2b",
                     "input": [{"text": instr, "image": "data:image/png;base64," + b64}]})


def ndcg_at_k(rank, k=10):
    return 1.0 / math.log2(rank + 2) if rank < k else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["sglang", "ref"], default="sglang")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-queries", type=int, default=0)
    ap.add_argument("--max-docs", type=int, default=0)
    a = ap.parse_args()

    corpus = load("origin_corpus.jsonl")
    queries = load("queries.jsonl")
    qrels = {r["query_id"]: r["relevant_docs"] for r in load("qrels.jsonl")}
    if a.max_docs:
        corpus = corpus[:a.max_docs]
    if a.max_queries:
        queries = queries[:a.max_queries]
    print("[data] docs=%d queries=%d" % (len(corpus), len(queries)))

    t0 = time.time()
    if a.backend == "ref":
        sys.path.insert(0, CKPT)
        from trident_embedder import TridentEmbedder
        m = TridentEmbedder(CKPT)
        instr_q, instr_d = m.instructions["query"], m.instructions["doc"]
        q_vecs = m.process([{"text": q["text"], "images": []} for q in queries], role="query")
        q_vecs = [v.tolist() for v in q_vecs]
        d_vecs = m.process([{"text": None, "images": [os.path.join(DATA, d["image"])]}
                            for d in corpus], role="doc")
        d_vecs = [v.tolist() for v in d_vecs]
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(CKPT, trust_remote_code=True)
        cfg = json.load(open(os.path.join(CKPT, "mmemb_model.json"), encoding="utf-8"))["prompt"]
        instr_q, instr_d = cfg["query_instruction"], cfg["doc_instruction"]

        def qp(text):
            conv = [{"role": "system", "content": [{"type": "text", "text": instr_q}]},
                    {"role": "user", "content": [{"type": "text", "text": text}]}]
            return tok.apply_chat_template(conv, tokenize=False,
                                           add_generation_prompt=cfg.get("add_generation_prompt", True))
        prompts = [qp(q["text"]) for q in queries]
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            q_vecs = list(ex.map(sglang_text, prompts))
        print("[sglang] queries done %.1fs" % (time.time() - t0))
        with ThreadPoolExecutor(max_workers=a.workers) as ex:
            d_vecs = list(ex.map(lambda d: sglang_image(os.path.join(DATA, d["image"]), instr_d),
                                 corpus))
    print("[embed] %.1fs total" % (time.time() - t0))

    import numpy as np

    Q = np.asarray(q_vecs, dtype="float32")
    D = np.asarray(d_vecs, dtype="float32")
    # 归一化后点积即余弦（官方输出已归一化，这里再兜一次，避免 bf16 舍入影响排序）
    Q /= (np.linalg.norm(Q, axis=1, keepdims=True) + 1e-12)
    D /= (np.linalg.norm(D, axis=1, keepdims=True) + 1e-12)
    np.savez_compressed(os.path.join(OUT_DIR, "emb_chartqa_%s.npz" % a.backend),
                        q=Q, d=D)

    ids = [d["id"] for d in corpus]
    ndcgs, r1, r5, r10 = [], 0, 0, 0
    for q, qv in zip(queries, Q):
        rel = set(qrels.get(q["id"], []))
        if not rel:
            continue
        sims = D @ qv
        order = np.argsort(-sims)
        rank = next((r for r, idx in enumerate(order) if ids[idx] in rel), None)
        if rank is None:
            ndcgs.append(0.0)
            continue
        ndcgs.append(ndcg_at_k(rank))
        r1 += rank < 1
        r5 += rank < 5
        r10 += rank < 10
    n = len(ndcgs)
    res = {"backend": a.backend, "dataset": "ChartQA/Origin", "docs": len(corpus),
           "queries": n, "ndcg@10": round(sum(ndcgs) / n * 100, 2),
           "recall@1": round(r1 / n * 100, 2), "recall@5": round(r5 / n * 100, 2),
           "recall@10": round(r10 / n * 100, 2), "seconds": round(time.time() - t0, 1)}
    os.makedirs(OUT_DIR, exist_ok=True)
    fn = os.path.join(OUT_DIR, "eval_chartqa_%s.json" % a.backend)
    json.dump(res, open(fn, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False))
    print("-> " + fn)


if __name__ == "__main__":
    main()
