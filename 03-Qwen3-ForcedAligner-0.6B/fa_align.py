#!/usr/bin/env python3
"""Qwen3-ForcedAligner-0.6B 在 HCU 上的强制对齐（vLLM in-process, pooling runner）。

算法复刻自官方 qwen_asr/inference/qwen3_forced_aligner.py：
  1. 文本分词 → "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
  2. 前置 "<|audio_start|><|audio_pad|><|audio_end|>"
  3. 模型对每个 <timestamp> 位置分类出时间格索引，timestamp_ms = idx * timestamp_segment_time(80)
  4. LIS 平滑（fix_timestamp）后按 (start, end) 成对切给每个词
"""
import json, os, sys

import numpy as np

MODEL = "/root/hygon/models/fa06"
TS_TOKEN_ID = 151705
SEG_MS = 80.0


def split_words(text):
    """与官方 tokenize_space_lang 一致的简化版：空格切分，CJK 逐字。"""
    out = []
    buf = []
    for ch in text:
        if ch.isspace():
            if buf:
                out.append("".join(buf)); buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out


def build_prompt(words):
    body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
    return "<|audio_start|><|audio_pad|><|audio_end|>" + body


def fix_timestamp(data):
    """官方 fix_timestamp：用最长递增子序列把异常跳变的格子拉回单调。"""
    data = [int(x) for x in data]
    n = len(data)
    if n == 0:
        return data
    dp = [1] * n
    parent = [-1] * n
    for i in range(1, n):
        for j in range(i):
            if data[j] <= data[i] and dp[j] + 1 > dp[i]:
                dp[i] = dp[j] + 1
                parent[i] = j
    idx = dp.index(max(dp))
    keep = []
    while idx != -1:
        keep.append(idx)
        idx = parent[idx]
    keep.reverse()
    out = list(data)
    # 非 LIS 位置用前一个已确认值填充，保证单调
    prev = 0
    ks = set(keep)
    for i in range(n):
        if i in ks:
            prev = out[i]
        else:
            out[i] = prev
    return out


def main(audio_path, text, language="English"):
    from vllm import LLM

    words = split_words(text)
    prompt = build_prompt(words)
    print(json.dumps({"words": words, "prompt_len": len(prompt)}, ensure_ascii=False))

    llm = LLM(
        model=MODEL,
        runner="pooling",
        hf_overrides={"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]},
        dtype="bfloat16",
        max_model_len=8192,
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        trust_remote_code=True,
    )
    import subprocess
    raw = subprocess.run(["ffmpeg", "-v", "quiet", "-i", audio_path,
                          "-f", "f32le", "-ac", "1", "-ar", "16000", "-"],
                         capture_output=True, check=True).stdout
    audio = (np.frombuffer(raw, dtype=np.float32).copy(), 16000)
    print("audio_sec", round(len(audio[0]) / 16000, 2), file=sys.stderr)
    outs = llm.encode(
        [{"prompt": prompt, "multi_modal_data": {"audio": audio}}],
        pooling_task="token_classify",
    )
    o = outs[0].outputs
    print("OUT_ATTRS", [a for a in dir(o) if not a.startswith("_")], file=sys.stderr)
    data = getattr(o, "data", None)
    print("DATA_TYPE", type(data), getattr(data, "shape", None), file=sys.stderr)
    arr = np.asarray(data)
    print("ARR", arr.shape, arr.dtype, file=sys.stderr)

    if arr.ndim >= 2:
        ids = arr.argmax(axis=-1)
    else:
        ids = arr.astype(np.int64)

    # 音频占位符会在引擎侧展开成多个 token，因此按「文本部分在序列尾部」对齐：
    # body 由 words 与 <timestamp> 交替组成，body_ids 一定是整条序列的后缀。
    tok = llm.get_tokenizer()
    body = "<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"
    body_ids = np.array(tok(body, add_special_tokens=False)["input_ids"])
    ts_mask = body_ids == TS_TOKEN_ID
    n_ts = int(ts_mask.sum())
    if len(ids) < len(body_ids):
        raise SystemExit(f"预测长度 {len(ids)} 小于文本 token 数 {len(body_ids)}")
    preds = ids[-len(body_ids):]
    ts_bins = preds[ts_mask]
    print("pred_tokens", len(ids), "body_tokens", len(body_ids), "ts_positions", n_ts,
          file=sys.stderr)
    assert len(ts_bins) == 2 * len(words), f"期望 {2*len(words)} 个时间戳，实得 {len(ts_bins)}"
    ts_ms = [int(x) * SEG_MS for x in ts_bins]
    fixed = fix_timestamp(ts_ms)
    items = []
    for i, w in enumerate(words):
        if 2 * i + 1 < len(fixed):
            items.append({"text": w,
                          "start": round(fixed[2 * i] / 1000.0, 3),
                          "end": round(fixed[2 * i + 1] / 1000.0, 3)})
    payload = {"audio": os.path.basename(audio_path), "language": language,
               "text": text, "items": items}
    print(json.dumps(payload, ensure_ascii=False, indent=1))
    out = os.environ.get("OUT_JSON")
    if out:
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "English")
