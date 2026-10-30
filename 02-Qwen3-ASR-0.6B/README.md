# Qwen3-ASR-0.6B on K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0

> 按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> 卡型说明：实测实例为 `K500SM_AI`，按赛事口径计入 `K100-AI`。
> 赛题归属：T04 语音（语音识别）。与 `Qwen3-ASR-1.7B` 同族，0.6B 为更小规格，按「同系列多参数规格各自单件计」独立计件。

## 1. 简介

本文档记录 `Qwen/Qwen3-ASR-0.6B` 在 `K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0`（精度 `bf16`）的适配与验证结果。

Qwen3-ASR-0.6B 是 Qwen3-ASR 系列的小规格语音识别模型，音频经音塔编码后由语言模型输出转写文本，支持多语种，单卡部署、显存占用低。

与官方部署方式的核心差异：未升级任何包、未改一行模型代码；运行时沿用镜像自带 DTK 与 Python 包，仅需两处装配（符号链接改指向、`ROCM_PATH` 指向镜像内 DTK），见第 3 节。

- 权重下载：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-0.6B
- 镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`
- 提交仓库：【待填】

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` |
| 卡型 | K500SM_AI 64 GB × 1（计 K100-AI），120 SM |
| 节点 | SCNet Notebook，Hygon C86 7185 32C / 503 GB RAM |
| Python / torch | 3.10.21 / 2.10.0+das.opt1.dtk2604.2608141812（HIP 6.3.26113） |
| triton / vllm | 3.6.0+das185.dtk2604 / 0.21.0+das.dtk2604 |
| transformers | 5.5.0 |
| 模型路径 | `/root/hygon/models/asr06` |
| 端口 | 8000 |

完整原始输出见 [`evidence/env.log`](evidence/env.log)。

## 3. 适配优化方案

### 3.1 基础镜像
`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`。

### 3.2 软件包升级
无。`requirements-extra.txt` 为空。

### 3.3 代码改动
无模型代码改动，仅两处运行时装配（不影响精度）：
1. `rootfs` 内 `vllm/.../fused_moe/modular_kernel.py` 为指向镜像绝对路径的符号链接，解包后失效 → 改指向 `$ROOTFS`（`run.sh` 幂等执行）。
2. triton 的 HCU 后端需 `ROCM_PATH` 下存在 `aillvm`，宿主 DTK 无该目录 → `ROCM_PATH` 指向镜像内 DTK。

### 3.4 升级步骤
解包镜像 → `bash run.sh` → `bash env_check.sh` → `bash smoke.sh` → 精度与性能。

## 4. 服务启动脚本

关键量：`CARD=K100-AI`、`TP=1`、`MAX_LEN=32768`、`MEM_FRAC=0.90`、`PORT=8000`。

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/hygon/models/asr06 \
  --served-model-name qwen3-asr-0.6b \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 8000
```

参数理由：`--dtype bfloat16` 与官方权重一致、匹配镜像内 flash-attn 路径；`--max-model-len 32768` 为原生上下文；
`--gpu-memory-utilization 0.90` 单卡独占留 10% 给运行时；未开 `--enforce-eager`（默认走 torch.compile + CUDA Graph，
首次启动一次性编译，缓存复用）；未设置 `NCCL_MIN_NCHANNELS`/`NCCL_MAX_NCHANNELS`。

## 5. Smoke 测试

`bash smoke.sh`，原始输出见 [`evidence/smoke.log`](evidence/smoke.log)：

```
== /health
HTTP 200
== /v1/models
{"object":"list","data":[{"id":"qwen3-asr-0.6b", ... "max_model_len":32768 ...
== ASR zh.mp3
{"text":"開放時間：早上九點至下午五點。","usage":{"type":"duration","seconds":6}}
== ASR en.mp3
{"text":"The tribal chief then called for the boy and presented him with fifty pieces of gold.","usage":{"type":"duration","seconds":8}}
```

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@zh.mp3" -F "model=qwen3-asr-0.6b"
```

中英两条均为正确转写。与 1.7B 相比，英文输出少一个逗号（`boy and presented` vs `boy, and presented`），
属模型规格差异，不影响语义。

## 6. 精度测评

**未完成 —— 缺官方输入。** 赛事要求 `verify.py` 按 `baselines/Qwen__Qwen3-ASR-0.6B.json` 写死的子集、
题数、解码参数与 GPU 基线分现场跑分；该基线文件与 `data/` 子集随分题材料下发，本仓库未持有，故不提供精度数字。
`verify.py` 已实现官方判分逻辑（容差 2%、只罚劣化、`--selfcheck` 通过），放入官方材料即可直接产出 `verify_result.json`。
按赛事红线不臆造分数。

## 7. 性能参考

官方写明 T04 测法随分题材料下发，故用自定口径（RTF = 墙钟 / 音频时长），与官方口径不同。
原始输出见 [`evidence/bench.txt`](evidence/bench.txt)。测试集同 1.7B（6 s 中文、8 s 英文、6 s 其他），全部 200。

| 并发 | 请求数 | 成功 | 墙钟 | 音频总长 | RTF | 吞吐 | QPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 63 | 63 | 11.34 s | 420 s | 0.027× | 37.04 audio-s/s | 5.56 |
| 8 | 48 | 48 | 2.51 s | 320 s | 0.008× | 127.26 audio-s/s | 19.09 |
| 32 | 96 | 96 | 2.84 s | 640 s | 0.004× | 225.05 audio-s/s | 33.76 |

对照 1.7B（21.97 / 119.53 / 208.23 audio-s/s）：0.6B 在单并发下快 1.7 倍，高并发下差距收敛，
说明并发上到 32 后瓶颈不在模型算力侧，而在服务链路。

## 8. 注意事项

- 必须用非 `-hf` 权重（`Qwen/Qwen3-ASR-0.6B`）；`-hf` 版额外带 `model.` 前缀，vLLM 0.21 加载会报
  `no module or parameter named 'model'`（1.7B 上已实测复现）。
- `ROCM_PATH` 必须是镜像内 DTK（含 `aillvm`），否则 kernel 编译报 `HSACOError ... .amdgcn`。
- 首次启动含 torch.compile，约 2–5 分钟，不是卡住。
- 纯文本请求（无音频）会返回无意义输出，判接口可用可以，不能当文本模型用。

---

### 仓库内容

| 文件 | 章节 | 说明 |
| --- | --- | --- |
| `env.sh` | 2 / 4 | 变量表 |
| `run.sh` | 4 | 装配环境 + 起服务 + 轮询 /health |
| `env_check.sh` | 2 | → `evidence/env.log` |
| `smoke.sh` | 5 | → `evidence/smoke.log` |
| `verify.py` | 6 | 官方判分逻辑（缺基线不臆造） |
| `bench.sh` / `asr_bench.py` | 7 | → `evidence/bench.txt` |
