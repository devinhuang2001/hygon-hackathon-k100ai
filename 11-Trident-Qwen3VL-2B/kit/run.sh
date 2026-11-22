#!/bin/bash
# Trident-Qwen3VL-2B × BW(gfx936) × SGLang 0.5.18
# 运行时：宿主 /usr/bin/python3.10 + 镜像 rootfs 里的 dist-packages + 镜像 DTK 用户态
set -u
R=/root/hygon/bundle/sglang/rootfs
export ROCM_PATH=$R/opt/dtk
export DTKROOT=$ROCM_PATH
export HIP_PATH=$ROCM_PATH/hip
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$ROCM_PATH/hip/bin:/opt/hyhal/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:$R/opt/dtk/.hyhal/rocm_smi/lib:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}
export PYTHONPATH=$R/usr/local/lib/python3.10/dist-packages:$R/usr/lib/python3/dist-packages
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1

# 宿主没有 python3.10-dev：Triton 现场编译 hip_utils.c 需要镜像里的 Python 头文件，
# 否则 Scheduler 直接 CalledProcessError（fatal error: Python.h: No such file or directory）
PH=/root/hygon/trident/python-headers
mkdir -p "$PH/x86_64-linux-gnu"
ln -sfnT "$R/usr/include/x86_64-linux-gnu/python3.10" "$PH/x86_64-linux-gnu/python3.10"
export C_INCLUDE_PATH="$R/usr/include/python3.10:$PH${C_INCLUDE_PATH:+:$C_INCLUDE_PATH}"
export CPLUS_INCLUDE_PATH="$R/usr/include/python3.10:$PH${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"
exec /usr/bin/python3.10 -m sglang.launch_server \
  --model-path /root/hygon/models/Trident-Qwen3VL-2B \
  --served-model-name trident-qwen3vl-2b \
  --is-embedding \
  --enable-multimodal \
  --json-model-override-args '{"architectures": ["Qwen3VLForConditionalGeneration"]}' \
  --host 127.0.0.1 --port 19100 \
  --trust-remote-code \
  --dtype bfloat16 \
  --attention-backend triton \
  --disable-cuda-graph \
  --mem-fraction-static 0.30
