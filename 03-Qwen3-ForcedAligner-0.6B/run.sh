#!/usr/bin/env bash
# 装配环境 + relink + 批跑官方 30 条强制对齐
set -o pipefail
cd "$(dirname "$0")"
source ./env.sh

# rootfs 符号链接修复（解包镜像后 vllm_hcu 的绝对链接失效，幂等）
python3 - <<'PYEOF'
import os
R = os.environ.get("ROOTFS", "/root/hygon/rootfs")
fix = 0
for base in (R + "/usr/local/lib/python3.10/dist-packages", R + "/usr/lib/python3/dist-packages"):
    for dp, dn, fn in os.walk(base):
        for n in dn + fn:
            p = os.path.join(dp, n)
            if os.path.islink(p):
                t = os.readlink(p)
                if t.startswith("/") and not t.startswith(R) and os.path.exists(R + t):
                    os.unlink(p); os.symlink(R + t, p); fix += 1
print("relinked", fix)
PYEOF

mkdir -p "$OUTDIR" evidence
python3 run_fa.py 2>&1 | tee evidence/fa_batch.log
echo "[run] done: $(ls "$OUTDIR" | wc -l) outputs"
