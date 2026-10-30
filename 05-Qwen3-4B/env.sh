#!/usr/bin/env bash
# 模型、镜像、框架、卡型、并行方式与启动参数，其余脚本读这里的变量
export MODEL_ID="Qwen/Qwen3-4B"
export MODEL_PATH="/root/public_data/model/admin/Qwen/Qwen3-4B"
export SERVED_NAME="qwen3-4b"
export IMAGE="harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10"
export FRAMEWORK="vllm"                                     # 推理框架：vllm | sglang
export CARD="K100-AI"                                       # 本实例为 K500SM_AI，按 K100-AI 计
export PRECISION="bf16"                                     # bf16 | int8 | int4
export PORT=8000

export TP=1                                                 # 单卡
export PP=1
export MAX_LEN=32768
export MEM_FRAC=0.90
export NNODES=1
export NODE_RANK=0
export LOAD_TIMEOUT=1800
export EXTRA_ARGS=""                                        # 例如 --enforce-eager

# 本实例无容器运行时，镜像经 harbor 拉取后在 Notebook 内解包为 rootfs
export ROOTFS="/root/hygon/rootfs"
