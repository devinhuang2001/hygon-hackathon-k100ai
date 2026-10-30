#!/usr/bin/env python3
"""verify.py —— 统一判分入口。两种模式：

1) 官方口径（默认）：读 `baselines/<模型>.json`，按文件里写死的子集 / 解码参数 / 官方 GPU 基线分，
   跑完用 `check()` 判「劣化是否 ≤ 2%」。基线文件随分题材料下发，本仓库未持有 → 缺文件时直接退出。
2) 自测绝对分（`--abs`）：不比对基线，用公开数据集按写死的配置实跑，只报绝对分与二项标准误，
   供验收方与官方基线人工比对。缺官方材料时用这个，不臆造任何基线分。

基线文件 schema（拿到官方文件后按此对齐即可跑模式 1）：
  {
    "dataset": "gsm8k",                       # gsm8k / mmlu
    "data": "data/gsm8k_test.jsonl",          # 子集（相对本目录）
    "train": "data/gsm8k_train.jsonl",        # few-shot exemplar，可省略
    "config": {"shots": 5, "max_tokens": 512},
    "metrics": {"accuracy": {"ref": 88.0, "scale": 100}}
  }
    - 越低越好的指标（wer / cer / aas）在 metric 里加 `"lower_is_better": true`
    - `scale` 缺省取 `ref`，容差 = 2% × scale

用法：
  python3 verify.py --selfcheck                                # 只验判分逻辑
  python3 verify.py --abs --dataset gsm8k --data data/gsm8k_test.jsonl \
                    --train data/gsm8k_train.jsonl --shots 5 --conc 24 --out r.json
  python3 verify.py --abs --dataset mmlu  --data data/mmlu_test.parquet --per-subject 5
  python3 verify.py                                            # 官方口径（需 baselines/）

注意：本文件内置的评测器（GSM8K / MMLU）是为了让 `--abs` 能真跑；官方评测器与评测子集若另行下发，
应以官方实现为准，本文件只保证判分容差逻辑与官方要求一致。
"""
import argparse
import json
import os
import re
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

TOL = 0.02  # 统一容差：劣化不超过 2%；有满分的按满分计（百分制即 2 pt），否则按基线值计

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("SERVED_NAME", "")     # OpenAI 接口里的 model 名
MODEL_ID = os.environ.get("MODEL_ID", "")     # 官方模型名，用于定位 baselines/<id>.json

PROMPT_GSM8K = "Question: {q}\nAnswer: Let's think step by step."
PROMPT_MMLU = ("Answer the following multiple choice question. "
               "Reply with the single letter of the correct option.\n\n"
               "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")


# ---------------------------------------------------------------- 判分逻辑

def check(got, m):
    """按基线条目判分。越低越好的指标在基线里标 lower_is_better。"""
    worse = got - m["ref"] if m.get("lower_is_better") else m["ref"] - got
    tol = TOL * m.get("scale", m["ref"])
    return {"got": round(got, 2), "ref": m["ref"], "worse": round(worse, 2),
            "tol": round(tol, 2), "pass": worse <= tol + 1e-9}


def selfcheck():
    """逻辑自检：劣化超容差必须 FAIL，容差内与提升必须 PASS。"""
    acc = {"ref": 80.0, "scale": 100}
    wer = {"ref": 5.0, "scale": 100, "lower_is_better": True}
    assert not check(77.9, acc)["pass"], "掉 2.1 pt 必须 FAIL"
    assert check(78.0, acc)["pass"] and check(85.0, acc)["pass"], "掉 2 pt 以内或涨点必须 PASS"
    assert not check(7.1, wer)["pass"] and check(5.0, wer)["pass"], "WER 升超容差 FAIL，不变 PASS"
    assert check(4.0, wer)["pass"], "WER 下降必须 PASS"
    print("selfcheck OK")


# ---------------------------------------------------------------- 推理与判分

def chat(prompt, max_tokens):
    import urllib.request
    body = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"]["content"] or ""


def last_int(text):
    """GSM8K 答案抽取：优先 #### 标记，其次 answer is N，兜底取最后一个整数。"""
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return int(m.group(1).replace(",", ""))
    m = re.search(r"(?:answer|result|total)\D{0,20}(-?[\d,]+)", text, re.I)
    if m:
        return int(m.group(1).replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*", text.replace("$", ""))
    return int(nums[-1].replace(",", "")) if nums else None


def first_letter(text):
    """MMLU 答案抽取：优先独立的 A/B/C/D，兜底第一个出现的字母。"""
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text.strip().upper())
    return m.group(1) if m else None


def load_gsm8k(path, limit=0, train=None, shots=0):
    pre = ""
    if shots and train:
        with open(train, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= shots:
                    break
                r = json.loads(line)
                pre += f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
        print(f"[verify] few-shot: {shots} exemplars from {os.path.basename(train)}", flush=True)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            items.append((pre + PROMPT_GSM8K.format(q=r["question"]), int(gold)))
    return items[:limit] if limit else items


def load_mmlu(path, per_subject=5):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("读 parquet 需要 pyarrow（官方镜像内自带），或先把子集转成 jsonl")
    t = pq.read_table(path).to_pydict()
    bucket = defaultdict(list)
    for i in range(len(t["question"])):
        subj = t["subject"][i]
        if len(bucket[subj]) >= per_subject:
            continue
        ch = t["choices"][i]
        ans = t["answer"][i]
        gold = ans if isinstance(ans, str) else "ABCD"[int(ans)]
        bucket[subj].append((PROMPT_MMLU.format(q=t["question"][i], a=ch[0], b=ch[1], c=ch[2], d=ch[3]),
                             gold))
    return [x for subj in sorted(bucket) for x in bucket[subj]]


def run_eval(dataset, items, max_tokens, conc, out, baseline=None, note=None):
    if not MODEL:
        sys.exit("请先 `source env.sh`（需要 SERVED_NAME）")
    print(f"[verify] {dataset}: {len(items)} items, conc={conc}, max_tokens={max_tokens}", flush=True)
    t0 = time.time()
    done = [0]
    results = [None] * len(items)

    def work(arg):
        i, (prompt, gold) = arg
        try:
            text = chat(prompt, max_tokens)
        except Exception as e:  # 单题失败不中断整轮
            results[i] = (False, None, "ERR " + str(e)[:120])
            done[0] += 1
            return
        pred = last_int(text) if dataset == "gsm8k" else first_letter(text)
        results[i] = (pred == gold, pred, text[-160:])
        done[0] += 1
        if done[0] % 25 == 0:
            ok = sum(1 for r in results if r and r[0])
            print(f"[verify] {done[0]}/{len(items)} acc={ok / done[0]:.4f} {time.time() - t0:.0f}s",
                  flush=True)

    with ThreadPoolExecutor(max_workers=conc) as ex:
        list(ex.map(work, enumerate(items)))

    n = len(results)
    correct = sum(1 for r in results if r[0])
    acc = correct / n if n else 0.0
    stderr = (acc * (1 - acc) / n) ** 0.5 if n else 0.0
    verdict = None
    if baseline:
        verdict = {k: check(acc, m) for k, m in baseline.get("metrics", {}).items()}
    res = {
        "model": MODEL, "model_id": MODEL_ID, "dataset": dataset,
        "n": n, "correct": correct, "accuracy": round(acc, 4), "stderr": round(stderr, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "mode": "official-baseline" if baseline else "abs",
        "verdict": verdict,
        "baseline": baseline.get("metrics") if baseline else None,
        "note": note or ("官方 baselines/<模型>.json 未随分题材料下发，此处为公开数据集上的绝对精度；"
                         "拿到官方基线文件后按同一配置复跑即可计算劣化幅度。"),
    }
    with open(out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    # 抽样落盘，便于人工核对判分是否被格式问题污染
    with open(out + ".samples.txt", "w", encoding="utf-8") as f:
        for (prompt, gold), (ok, pred, tail) in list(zip(items, results))[:8]:
            f.write(f"gold={gold} pred={pred} ok={ok}\n  out_tail={tail!r}\n")
    print(json.dumps({k: res[k] for k in ("dataset", "n", "correct", "accuracy", "stderr",
                                          "elapsed_s", "mode")}, ensure_ascii=False))
    return res


# ---------------------------------------------------------------- 入口

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="只验证判分逻辑")
    ap.add_argument("--abs", action="store_true", help="自测绝对分模式（不比对基线）")
    ap.add_argument("--dataset", choices=["gsm8k", "mmlu"])
    ap.add_argument("--data", default="")
    ap.add_argument("--train", default="")
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--per-subject", type=int, default=5)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="verify_result.json")
    a = ap.parse_args()

    if a.selfcheck:
        selfcheck()
        return

    base_path = f"baselines/{MODEL_ID.replace('/', '__')}.json" if MODEL_ID else ""
    baseline = None
    if base_path and os.path.exists(base_path):
        with open(base_path, encoding="utf-8") as f:
            baseline = json.load(f)
        # 基线文件优先决定子集与解码参数
        a.dataset = a.dataset or baseline.get("dataset")
        a.data = a.data or baseline.get("data", "")
        a.train = a.train or baseline.get("train", "")
        a.shots = baseline.get("config", {}).get("shots", a.shots)
        a.max_tokens = a.max_tokens or baseline.get("config", {}).get("max_tokens", 0)
    elif not a.abs:
        sys.exit(f"缺少官方基线文件 {base_path or 'baselines/<模型>.json'}（随分题材料下发，"
                 f"需向赛事方获取），无法按官方口径判分。\n"
                 f"自测绝对分请加 `--abs`；判分逻辑可用 `--selfcheck` 验证。")

    if not a.dataset or not a.data:
        sys.exit("需要 --dataset 与 --data")
    if not os.path.exists(a.data):
        sys.exit(f"数据不存在：{a.data}")

    max_tokens = a.max_tokens or (512 if a.dataset == "gsm8k" else 32)
    items = (load_gsm8k(a.data, a.limit, a.train, a.shots) if a.dataset == "gsm8k"
             else load_mmlu(a.data, a.per_subject))
    run_eval(a.dataset, items, max_tokens, a.conc, a.out, baseline=baseline)


if __name__ == "__main__":
    main()
