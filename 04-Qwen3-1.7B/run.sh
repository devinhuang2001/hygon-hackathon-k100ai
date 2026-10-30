#!/usr/bin/env bash
# 模型启动脚本：准备镜像栈 → 拉起 vLLM serving → 轮询 /health 就绪
# 本实例（SCNet Notebook）没有容器运行时，镜像层解包在 $ROOTFS；驱动 /opt/hyhal 由平台注入
set -euo pipefail
cd "$(dirname "$0")"
source ./env.sh

# 1) 组装运行时环境：包与 DTK 取自镜像 rootfs，驱动取自宿主
source /opt/dtk/env.sh
export ROCM_PATH="$ROOTFS/opt/dtk"          # 必须指向镜像内 DTK，triton 需要其中的 aillvm
export DTKROOT="$ROCM_PATH"
export HIP_PATH="$ROCM_PATH/hip"
export LD_LIBRARY_PATH="$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:/opt/hyhal/lib:/opt/hyhal/lib64:${LD_LIBRARY_PATH:-}"
export PATH="$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$ROCM_PATH/aillvm/bin:$ROCM_PATH/hip/bin:/opt/hyhal/bin:$PATH"
export PYTHONPATH="$ROOTFS/usr/local/lib/python3.10/dist-packages:$ROOTFS/usr/lib/python3/dist-packages"

# 2) 镜像内 vllm_hcu 用绝对符号链接覆盖 vllm 的部分模块，解包后需改指向 rootfs
python3 - <<'PY'
import os
R = os.environ["ROOTFS"]
fix = 0
for base in (f"{R}/usr/local/lib/python3.10/dist-packages", f"{R}/usr/lib/python3/dist-packages"):
    for dp, dn, fn in os.walk(base):
        for n in dn + fn:
            p = os.path.join(dp, n)
            if os.path.islink(p):
                t = os.readlink(p)
                if t.startswith("/") and not t.startswith(R) and os.path.exists(R + t):
                    os.unlink(p); os.symlink(R + t, p); fix += 1
print(f"[run] relinked {fix} absolute symlink(s)")
PY

# 3) 拉起服务
CMD=(python3 -m vllm.entrypoints.openai.api_server
     --model "$MODEL_PATH" --served-model-name "$SERVED_NAME"
     --trust-remote-code --dtype bfloat16
     --max-model-len "$MAX_LEN" --gpu-memory-utilization "$MEM_FRAC"
     --host 127.0.0.1 --port "$PORT")
echo "[run] ${CMD[*]} $EXTRA_ARGS" | tee server.log
"${CMD[@]}" $EXTRA_ARGS >> server.log 2>&1 &
SERVER_PID=$!

# 4) 轮询 /health（不用固定 sleep；进程提前退出立即报错）
for ((t = 0; t < LOAD_TIMEOUT; t += 5)); do
  if ! kill -0 "$SERVER_PID" 2>/dev/null; then
    echo "[run] server exited, see server.log" >&2; exit 1
  fi
  if curl -sf --noproxy "*" "http://127.0.0.1:$PORT/health" > /dev/null; then
    echo "[run] server up after ${t}s"; exit 0
  fi
  sleep 5
done
echo "[run] server not ready in ${LOAD_TIMEOUT}s, see server.log" >&2
kill "$SERVER_PID"; exit 1
