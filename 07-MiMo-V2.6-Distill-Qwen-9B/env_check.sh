#!/usr/bin/env bash
# 采集验证环境写入 env.log；报告「2. 验证环境」的版本表直接取自这里
# 在 run.sh 之后执行，记录的是升级 / 打 patch 之后实际生效的版本
source /root/hygon/env_img.sh
{
  echo "image: harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10"
  echo "image_note: 镜像层在 Notebook 内解包到 /root/hygon/rootfs（本实例无容器运行时），DTK/包取自镜像，驱动 /opt/hyhal 由平台注入"
  echo "dtk: $(cat /opt/dtk/.info/version 2>/dev/null || readlink -f /opt/dtk)  (rocm_path=$ROCM_PATH)"
  hy-smi
  python -c "import sys, torch; print('python:', sys.version.split()[0]); print('torch:', torch.__version__); print('HIP:', torch.version.hip); print('Device:', torch.cuda.get_device_name(0), 'x', torch.cuda.device_count())"
  pip list 2>/dev/null | grep -iE "^(torch|triton|transformers|tokenizers|sglang|vllm|flash|fla|lmslim|ray)|das"
} 2>&1 | tee /root/hygon/env.log
