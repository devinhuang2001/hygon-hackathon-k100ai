#!/usr/bin/env bash
# 性能测试：官方 6 档（4K/16K 输入 × 1K 输出 × 并发 1/8/64），每档 2 轮，第 2 轮计入
# 与官方模板的差异（已在报告第 8 节说明）：
#   1) 官方模板用 sglang.benchmark.serving 客户端；本镜像不含 sglang，改用 vLLM 自带 `vllm bench serve`。
#   2) 官方写 --random-range-ratio 1；vLLM 客户端在该取值下直接报错（min input tokens 为 0），故用 0.99。
#   3) 加 --ignore-eos，保证输出长度固定为 1024，与「1K 输出」口径一致。
set -uo pipefail
cd "$(dirname "$0")"
H=/root/hygon
source ./env.sh
source /opt/dtk/env.sh
export ROCM_PATH="$H/rootfs/opt/dtk"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$H/rootfs/usr/local/lib/python3.10/dist-packages:$H/rootfs/usr/lib/python3/dist-packages"

mkdir -p bench; rm -f bench/round*.jsonl bench/round*.json
for ILEN in 4096 16384; do
  for C in 1 8 64; do
    N=$(( C * 2 > 8 ? C * 2 : 8 ))
    for R in 1 2; do
      TAG="round${R}_${ILEN}_1024_c${C}"
      echo "[bench] $TAG"
      timeout 1200 python3 -m vllm.entrypoints.cli.main bench serve \
        --backend openai --host 127.0.0.1 --port "$PORT" --model "$SERVED_NAME" \
        --tokenizer "$MODEL_PATH" --dataset-name random \
        --random-input-len "$ILEN" --random-output-len 1024 \
        --random-range-ratio 0.99 --ignore-eos \
        --max-concurrency "$C" --num-prompts "$N" --seed 1 \
        --save-result --result-dir bench --result-filename "$TAG.json" \
        > "bench/${TAG}.log" 2>&1 || echo "[bench] FAIL $TAG（写进报告注意事项）"
    done
  done
done

python3 - <<'PY'
import json, glob, os, re
rows = []
for f in glob.glob("bench/round2_*.json"):
    m = re.search(r"round\d_(\d+)_(\d+)_c(\d+)", os.path.basename(f))
    rows.append((int(m.group(1)), int(m.group(3)), json.load(open(f))))
rows.sort(key=lambda x: (x[0], x[1]))
cols = ["completed","duration","request_throughput","output_throughput",
        "mean_ttft_ms","p99_ttft_ms","mean_tpot_ms","p99_tpot_ms"]
with open("bench/summary.md", "w") as o:
    o.write("| input_len | concurrency | " + " | ".join(cols) + " |\n")
    o.write("|" + " --- |" * (len(cols) + 2) + "\n")
    for ilen, c, d in rows:
        cells = [f"{d.get(k):.2f}" if isinstance(d.get(k), float) else str(d.get(k, "-")) for k in cols]
        o.write(f"| {ilen} | {c} | " + " | ".join(cells) + " |\n")
print(open("bench/summary.md").read())
PY
