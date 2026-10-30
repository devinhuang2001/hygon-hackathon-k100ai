#!/usr/bin/env bash
# Tencent-Hunyuan/AuK · K100-AI（K500SM_AI）环境变量表
export AUK_PATH=${AUK_PATH:-/root/hygon/models/auk}
export QWEN_PATH=${QWEN_PATH:-/root/hygon/models/qwen-omni}
export AUK_SRC=${AUK_SRC:-/root/hygon/work/AuK/src}
export ROOTFS=${ROOTFS:-/root/hygon/rootfs}
export OUTDIR=${OUTDIR:-evidence/formal_auk/outputs}
export COND=${COND:-formal/auk.jsonl}
export PYTHONPATH=$AUK_SRC:/root/hygon/pydeps:$ROOTFS/usr/local/lib/python3.10/dist-packages:$PYTHONPATH
