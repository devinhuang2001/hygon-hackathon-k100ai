#!/usr/bin/env bash
# Smoke 测试：健康检查 + 模型列表 + ASR 转写 + 纯文本问答
# env.sh 里有未定义变量，source 之后不能再开 -u

source ./env.sh
URL="http://127.0.0.1:$PORT"
CURL=(curl -sS --noproxy "*" --max-time 300)
exec > >(tee smoke.log) 2>&1

echo "== /health";      "${CURL[@]}" -o /dev/null -w "HTTP %{http_code}\n" "$URL/health"
echo "== /v1/models";   "${CURL[@]}" "$URL/v1/models" | head -c 300; echo
echo "== ASR 中文 (zh.mp3)"
"${CURL[@]}" -X POST "$URL/v1/audio/transcriptions" -F "file=@data/zh.mp3" -F "model=$SERVED_NAME"; echo
echo "== ASR 英文 (en.mp3)"
"${CURL[@]}" -X POST "$URL/v1/audio/transcriptions" -F "file=@data/en.mp3" -F "model=$SERVED_NAME"; echo
echo "== 纯文本问答 (无音频，仅验接口)"
"${CURL[@]}" "$URL/v1/completions" -H "Content-Type: application/json" \
  -d "{\"model\":\"$SERVED_NAME\",\"prompt\":\"hello\",\"max_tokens\":8}" | head -c 200; echo
echo "== 完成"
