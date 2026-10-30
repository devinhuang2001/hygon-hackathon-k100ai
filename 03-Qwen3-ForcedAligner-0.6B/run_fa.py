#!/usr/bin/env python3
"""Qwen3-ForcedAligner-0.6B 批量对齐 T04 条件清单（vLLM in-process, pooling）。
算法复刻官方 qwen_asr: "<timestamp><timestamp>".join(words) → token 分类 → LIS 平滑。
输出: outputs/forced_aligner/<id>.json  （单词起止时间, 单位: 秒）
"""
import json, os, sys, glob
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
COND = os.path.join(BASE, "formal", "forced_aligner.jsonl")
OUTDIR = os.path.join(BASE, "evidence", "formal_fa", "outputs")
MODEL = "/root/hygon/models/fa06"
TS_TOKEN_ID = 151705
SEG_MS = 80.0


def split_words(text):
    out, buf = [], []
    for ch in text:
        if ch.isspace():
            if buf:
                out.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def fix_timestamp(data):
    data = [int(x) for x in data]
    n = len(data)
    if n == 0:
        return data
    dp, parent = [1] * n, [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i], parent[i] = dp[j] + 1, j
    idx = dp.index(max(dp))
    keep = []
    while idx != -1:
        keep.append(idx); idx = parent[idx]
    keep.reverse()
    out, prev, ks = list(data), 0, set(keep)
    for i in range(n):
        if i in ks:
            prev = out[i]
        else:
            out[i] = prev
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = [json.loads(l) for l in open(COND, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if not os.path.exists(os.path.join(OUTDIR, r["id"] + ".json"))]
    print(f"[fa] total={len(rows)} todo={len(todo)}", flush=True)
    if not todo:
        return
    from vllm import LLM
    llm = LLM(model=MODEL, runner="pooling",
              hf_overrides={"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]},
              dtype="bfloat16", max_model_len=8192, enforce_eager=True,
              gpu_memory_utilization=0.35, trust_remote_code=True)
    tok = llm.get_tokenizer()
    import subprocess
    for r in todo:
        rid = r["id"]
        # 官方 jsonl 的相对路径以 formal/conditions/ 为基准（../../data/...）
        audio_path = os.path.normpath(os.path.join(BASE, "formal", "conditions", r["assets"]["audio"]["path"]))
        text = r["text"]
        language = r["params"].get("language", "English")
        try:
            words = split_words(text)
            prompt = "<|audio_start|><|audio_pad|><|audio_end|>" + \
                     "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
            raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", audio_path,
                                  "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                                 capture_output=True, check=True).stdout
            audio = (np.frombuffer(raw, dtype=np.float32).copy(), 16000)
            outs = llm.encode([{"prompt": prompt, "multi_modal_data": {"audio": audio}}],
                              pooling_task="token_classify")
            arr = np.asarray(outs[0].outputs.data)
            ids = arr.argmax(axis=-1) if arr.ndim >= 2 else arr.astype(np.int64)
            body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
            body_ids = np.array(tok(body, add_special_tokens=False)["input_ids"])
            ts_mask = body_ids == TS_TOKEN_ID
            if len(ids) < len(body_ids):
                raise RuntimeError(f"pred {len(ids)} < body {len(body_ids)}")
            preds = ids[-len(body_ids):]
            ts_bins = preds[ts_mask]
            assert len(ts_bins) == 2 * len(words), f"{len(ts_bins)} != {2*len(words)}"
            fixed = fix_timestamp([int(x) * SEG_MS for x in ts_bins])
            items = [{"text": w, "start": round(fixed[2*i]/1000.0, 3),
                      "end": round(fixed[2*i+1]/1000.0, 3)}
                     for i, w in enumerate(words)]
            payload = {"condition_id": rid, "language": language, "unit": "seconds", "words": items}
            with open(os.path.join(OUTDIR, rid + ".json"), "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=1)
            print(f"[fa] OK {rid} words={len(items)}", flush=True)
        except Exception as e:
            print(f"[fa] FAIL {rid}: {e}", flush=True)
            with open(os.path.join(OUTDIR, rid + ".error.txt"), "w", encoding="utf-8") as f:
                f.write(str(e))


if __name__ == "__main__":
    main()
