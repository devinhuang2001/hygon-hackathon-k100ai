#!/usr/bin/env bash
# 装配环境 + 批跑官方 120 条零样本 TTS（voice clone）
set -o pipefail
cd "$(dirname "$0")"
source ./env.sh

mkdir -p "$OUTDIR" evidence
python3 run_auk.py 2>&1 | tee evidence/auk_batch.log
echo "[run] done: $(ls "$OUTDIR" | wc -l) outputs"
