#!/usr/bin/env bash
# 性能测试：ASR 口径（RTF / audio-s per second / QPS），并发 1 / 8 / 32
# 赛事文档写明「生成类与语音类的测法随分题材料下发」，官方未下发 T04 压测脚本，
# 因此这里用自定但可复现的口径，并注明与官方口径不同。
set -uo pipefail
cd "$(dirname "$0")"
source ./env.sh
source /opt/dtk/env.sh
export ROCM_PATH="$ROOTFS/opt/dtk"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"
export PYTHONPATH="$ROOTFS/usr/local/lib/python3.10/dist-packages:$ROOTFS/usr/lib/python3/dist-packages"

mkdir -p bench
python3 asr_bench.py 2>&1 | tee bench/asr_bench.txt
echo "[bench] done; 见 bench/asr_bench.txt"
