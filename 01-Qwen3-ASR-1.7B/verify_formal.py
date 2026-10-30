# -*- coding: utf-8 -*-
"""按官方条件清单对 qwen3_asr 300 条输出计算 CER/WER。"""
import json, os, re, sys

BASE = os.path.dirname(os.path.abspath(__file__))
COND = os.path.join(BASE, "formal/conditions/qwen3_asr.jsonl")
OUT = os.path.join(BASE, "evidence", "formal_asr", "outputs")

_PUNCT_ZH = re.compile(r"[，。！？；：、“”‘’（）…—\-\.,!?;:\"'()]+")
_PUNCT_EN = re.compile(r"[^a-z0-9' ]+")

def norm(s, lang):
    s = s.strip()
    if lang == "zh":
        return list(_PUNCT_ZH.sub("", s))
    s = s.lower()
    s = _PUNCT_EN.sub(" ", s)
    return s.split()

def edit(a, b):
    if a == b:
        return 0
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (a[i - 1] != b[j - 1]))
            prev = cur
    ins = dels = subs = None
    return dp[n]

def cer(ref, hyp):
    r, h = norm(ref, "zh"), norm(hyp, "zh")
    d = edit(r, h)
    return d, len(r)

def wer(ref, hyp):
    r, h = norm(ref, "en"), norm(hyp, "en")
    d = edit(r, h)
    return d, len(r)

stats = {"zh": [0, 0, 0], "en": [0, 0, 0], "clean": [0, 0, 0], "other": [0, 0, 0]}  # err, total, n
missing = []
rows = []
for line in open(COND, encoding="utf-8"):
    c = json.loads(line)
    rid = c["id"]
    lang = c["language"]
    p = os.path.join(OUT, rid + ".txt")
    if not os.path.exists(p):
        missing.append(rid)
        continue
    hyp = open(p, encoding="utf-8").read()
    ref = c["text"]
    if lang == "zh":
        e, t = cer(ref, hyp)
    else:
        e, t = wer(ref, hyp)
        bucket = "clean" if ":test-clean:" in c["source"] else "other"
        stats[bucket][0] += e
        stats[bucket][1] += t
        stats[bucket][2] += 1
    stats[lang][0] += e
    stats[lang][1] += t
    stats[lang][2] += 1
    rows.append((rid, lang, e, t))

res = {"missing": missing}
for lang, name in (("zh", "AISHELL-1 CER"), ("en", "LibriSpeech WER"), ("clean", "LibriSpeech test-clean WER"), ("other", "LibriSpeech test-other WER")):
    e, t, n = stats[lang]
    res[name] = {"errors": e, "tokens": t, "utterances": n, "rate": round(e / t * 100, 2) if t else None}
print(json.dumps(res, ensure_ascii=False, indent=1))
with open(os.path.join(BASE, "evidence", "formal_asr", "verify_result.json"), "w", encoding="utf-8") as f:
    json.dump(res, f, ensure_ascii=False, indent=1)
