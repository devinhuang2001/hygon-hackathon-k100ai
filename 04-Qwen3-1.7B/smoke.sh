#!/usr/bin/env bash
# 文本模型 smoke：健康检查 + 单轮 + 多轮 + 算术 + 流式（均关闭思考，便于看最终回答）
H=/root/hygon
OUT=${1:-$H/runs/text-qwen3-0.6b}/smoke.log
U=http://127.0.0.1:8000
M=qwen3-0.6b

chat() {
  curl -sS --noproxy "*" "$U/v1/chat/completions" -H "Content-Type: application/json" -d "$1" --max-time 300 \
  | python3 -c 'import sys,json
d=json.load(sys.stdin)
c=d["choices"][0]
print(repr(c["message"]["content"]))
print("finish:",c["finish_reason"],"tokens:",d["usage"]["completion_tokens"])'
}

{
  echo "== /health"
  curl -sS --noproxy "*" -o /dev/null -w "HTTP %{http_code}\n" --max-time 30 "$U/health"
  echo "== /v1/models"
  curl -sS --noproxy "*" --max-time 30 "$U/v1/models" | head -c 200; echo
  echo "== 单轮(关闭思考)"
  chat "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"用一句话介绍国产异构加速器。\"}],\"temperature\":0,\"max_tokens\":128,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  echo "== 多轮(关闭思考)"
  chat "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"我叫小海。\"},{\"role\":\"assistant\",\"content\":\"你好，小海！\"},{\"role\":\"user\",\"content\":\"我叫什么名字？\"}],\"temperature\":0,\"max_tokens\":128,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  echo "== 算术(关闭思考)"
  chat "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"13*17 等于多少？只回答数字。\"}],\"temperature\":0,\"max_tokens\":64,\"chat_template_kwargs\":{\"enable_thinking\":false}}"
  echo "== 流式"
  curl -sS --noproxy "*" "$U/v1/chat/completions" -H "Content-Type: application/json" \
    -d "{\"model\":\"$M\",\"messages\":[{\"role\":\"user\",\"content\":\"数到三。\"}],\"temperature\":0,\"max_tokens\":64,\"stream\":true,\"chat_template_kwargs\":{\"enable_thinking\":false}}" \
    --max-time 120 | tail -n 3
} > "$OUT" 2>&1
cat "$OUT"
