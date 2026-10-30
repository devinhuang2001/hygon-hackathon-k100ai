#!/usr/bin/env bash
# Qwen3-ForcedAligner-0.6B · K100-AI（K500SM_AI）· vLLM 0.21.0 环境变量表
export MODEL=${MODEL:-/root/hygon/models/fa06}
export ROOTFS=${ROOTFS:-/root/hygon/rootfs}
export MAX_LEN=${MAX_LEN:-8192}
export MEM_FRAC=${MEM_FRAC:-0.35}
export OUTDIR=${OUTDIR:-evidence/formal_fa/outputs}
export COND=${COND:-formal/forced_aligner.jsonl}
# 镜像内 DTK（triton HCU 后端需要 aillvm）
export ROCM_PATH=$ROOTFS/opt/dtk
export PATH=$ROCM_PATH/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ROOTFS/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
