#!/usr/bin/env python3
"""精度自测：对已起好的 vLLM 服务跑公开数据集，产出绝对精度 json。

**这不是 verify.py**。verify.py 是官方判分入口，必须读 `baselines/<模型>.json`（写死子集题数、
max_tokens、思考开关与官方 GPU 基线分）才能算「较基线劣化是否 ≤2%」；该文件随分题材料下发，
本仓库未持有，verify.py 在缺文件时直接退出（只保留 --selfcheck 验证判分逻辑）。
本脚本是补位的自测工具：不比对基线，只给绝对分，且把配置写死在下面，便于复现与横向比对。

配置（写死，勿随意改，改动需在报告第 6 节同步声明）：
  GSM8K：5-shot（train 前 5 条作 exemplar，含完整 CoT），greedy，max_tokens 512，
         答案优先取输出里的 `####`，其次 "answer is N"，兜底取最后一个整数
  MMLU ：0-shot，greedy，max_tokens 32，答案取输出中第一个 A/B/C/D，每科 5 题
  两者均关闭思考模式（`chat_template_kwargs.enable_thinking=false`），与官方「思考开关写死」的口径一致。

注意口径敏感：同一模型 GSM8K 若改成 0-shot 且不给 CoT 提示，实测只有 17.06%；
5-shot 后为 87.26%（本次实跑）。比对官方基线前必须先确认 shot 数是否一致。

用法：
  python3 precision.py --dataset gsm8k --data data/gsm8k_test.jsonl --train data/gsm8k_train.jsonl --shots 5
  python3 precision.py --dataset mmlu  --data data/mmlu_test.parquet --per-subject 5
  python3 precision.py --dataset gsm8k --data data/gsm8k_test.jsonl --train data/gsm8k_train.jsonl --shots 5 --limit 200 --conc 16
"""
import argparse, json, os, re, sys, time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("VLLM_URL", "http://127.0.0.1:8000")
MODEL = os.environ.get("SERVED_NAME", "")

PROMPT_GSM8K = "Question: {q}\nAnswer: Let's think step by step."
PROMPT_MMLU = ("Answer the following multiple choice question. "
               "Reply with the single letter of the correct option.\n\n"
               "Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer:")


def chat(prompt, max_tokens):
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
    # 优先取 GSM8K 标准答案标记 ####，其次 "answer is N"，兜底取最后一个整数
    m = re.search(r"####\s*(-?[\d,]+)", text)
    if m:
        return int(m.group(1).replace(",", ""))
    m = re.search(r"(?:answer|result|total)\D{0,20}(-?[\d,]+)", text, re.I)
    if m:
        return int(m.group(1).replace(",", ""))
    nums = re.findall(r"-?\d[\d,]*", text.replace("$", ""))
    if not nums:
        return None
    return int(nums[-1].replace(",", ""))


def first_letter(text):
    m = re.search(r"\b([ABCD])\b", text.strip().upper())
    if m:
        return m.group(1)
    m = re.search(r"([ABCD])", text.strip().upper())
    return m.group(1) if m else None


def load_gsm8k(path, limit, train=None, shots=0):
    pre = ""
    if shots and train:
        with open(train, encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= shots:
                    break
                r = json.loads(line)
                pre += f"Question: {r['question']}\nAnswer: {r['answer']}\n\n"
        print(f"[prec] few-shot: {shots} exemplars from {os.path.basename(train)}", flush=True)
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            gold = r["answer"].split("####")[-1].strip().replace(",", "")
            items.append((pre + PROMPT_GSM8K.format(q=r["question"]), int(gold)))
    return items[:limit] if limit else items


def load_mmlu(path, per_subject):
    from collections import defaultdict
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.exit("需要 pyarrow 读 parquet（镜像内 datasets 自带）")
    t = pq.read_table(path).to_pydict()
    bucket = defaultdict(list)
    for i in range(len(t["question"])):
        subj = t["subject"][i]
        if len(bucket[subj]) >= per_subject:
            continue
        ch = t["choices"][i]
        ans = t["answer"][i]
        gold = ans if isinstance(ans, str) else "ABCD"[int(ans)]
        bucket[subj].append((PROMPT_MMLU.format(q=t["question"][i], a=ch[0], b=ch[1], c=ch[2], d=ch[3]), gold))
    items = [x for subj in sorted(bucket) for x in bucket[subj]]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["gsm8k", "mmlu"], required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--per-subject", type=int, default=5)
    ap.add_argument("--conc", type=int, default=16)
    ap.add_argument("--max-tokens", type=int, default=0)
    ap.add_argument("--shots", type=int, default=0)
    ap.add_argument("--train", default="")
    ap.add_argument("--out", default="verify_result.json")
    a = ap.parse_args()

    if not MODEL:
        sys.exit("请先 `source env.sh`（需要 SERVED_NAME）")
    max_tokens = a.max_tokens or (512 if a.dataset == "gsm8k" else 32)
    items = (load_gsm8k(a.data, a.limit, a.train, a.shots) if a.dataset == "gsm8k"
             else load_mmlu(a.data, a.per_subject))
    print(f"[prec] {a.dataset}: {len(items)} items, conc={a.conc}, max_tokens={max_tokens}", flush=True)

    t0 = time.time()
    done = [0]
    results = [None] * len(items)

    def work(arg):
        i, (prompt, gold) = arg
        try:
            out = chat(prompt, max_tokens)
        except Exception as e:  # 单题失败不中断整轮
            results[i] = (False, None, "ERR " + str(e)[:120])
            done[0] += 1
            return
        pred = last_int(out) if a.dataset == "gsm8k" else first_letter(out)
        results[i] = (pred == gold, pred, out[-160:])
        done[0] += 1
        if done[0] % 25 == 0:
            ok = sum(1 for r in results if r and r[0])
            print(f"[prec] {done[0]}/{len(items)} acc={ok/done[0]:.4f} "
                  f"{time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=a.conc) as ex:
        list(ex.map(work, enumerate(items)))

    n = len(results)
    correct = sum(1 for r in results if r[0])
    acc = correct / n if n else 0.0
    # 二项分布标准误，便于判断与基线的差异是否在噪声内
    stderr = (acc * (1 - acc) / n) ** 0.5 if n else 0.0
    res = {
        "model": MODEL,
        "dataset": a.dataset,
        "n": n,
        "correct": correct,
        "accuracy": round(acc, 4),
        "stderr": round(stderr, 4),
        "elapsed_s": round(time.time() - t0, 1),
        "config": {
            "temperature": 0, "shots": a.shots, "max_tokens": max_tokens,
            "enable_thinking": False, "conc": a.conc,
            "per_subject": a.per_subject if a.dataset == "mmlu" else None,
            "source": os.path.basename(a.data),
        },
        "baseline": None,
        "note": ("官方 baselines/<模型>.json 未随分题材料下发，此处为公开数据集上的绝对精度；"
                 "拿到官方基线文件后按同一配置复跑即可计算劣化幅度。"),
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=2)
    # 抽样落盘，便于人工核对判分是否被格式问题污染
    with open(a.out + ".samples.txt", "w", encoding="utf-8") as f:
        for (prompt, gold), (ok, pred, tail) in list(zip(items, results))[:8]:
            f.write(f"gold={gold} pred={pred} ok={ok}\n  out_tail={tail!r}\n")
    print(json.dumps({k: res[k] for k in ("dataset", "n", "correct", "accuracy", "stderr", "elapsed_s")},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
