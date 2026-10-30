# Qwen3-ASR-1.7B on K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0

> 本报告按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> 卡型说明：实测实例为 `K500SM_AI`，按赛事口径计入 `K100-AI`。

## 1. 简介

本文档记录 `Qwen/Qwen3-ASR-1.7B` 在 `K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0`（精度 `bf16`）环境的适配与验证结果。

Qwen3-ASR-1.7B 是 Qwen 团队的语音识别模型，1.7B 参数，音频经音塔编码后由语言模型自回归输出转写文本，支持中英等多语种与语言识别，单卡即可部署。

与官方部署方式的核心差异：镜像内的 DTK 与 Python 包直接复用，未升级任何包、未改动一行模型代码；运维层面绕开了两个非模型问题——镜像内 `vllm_hcu` 的绝对符号链接在解包后需改指向 rootfs，且 triton 的 HCU 后端要求 `ROCM_PATH` 指向镜像内 DTK（其中的 `aillvm`）而非宿主 DTK。

相关获取地址：

- 权重下载：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B
- 镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`
- 提交仓库：https://github.com/devinhuang2001/hygon-hackathon-k100ai/tree/main/01-Qwen3-ASR-1.7B

参考文档：

- Qwen3-ASR-1.7B 官方模型卡（用法、限制与 GPU 参考值）：https://www.modelscope.cn/models/Qwen/Qwen3-ASR-1.7B
- vLLM OpenAI API Server 文档：https://docs.vllm.ai
- 官方分题材料（2026-10-30 下发）：条件清单 `formal/conditions/qwen3_asr.jsonl`（原件在仓库内）

一键复现：

```bash
git clone https://github.com/devinhuang2001/hygon-hackathon-k100ai && cd hygon-hackathon-k100ai/01-Qwen3-ASR-1.7B
bash run.sh                                    # 装配 rootfs + 起 vLLM 服务（127.0.0.1:8000），轮询 /health
bash env_check.sh                              # 采集版本 → evidence/env.log
bash smoke.sh                                  # 健康检查 + 中英转写 → evidence/smoke.log
source env.sh && python3 verify.py --selfcheck # 确认官方判分逻辑未被改动
python3 evidence/formal_asr/run_asr.py         # 官方 300 条转写 → evidence/formal_asr/outputs/
python3 verify_formal.py                       # CER/WER 打分 → evidence/formal_asr/verify_result.json
bash bench.sh                                  # ASR 并发吞吐 → evidence/bench.txt
```

说明：`run_asr.py` 需分题材料的音频资产（未随仓库分发，按条件清单 `assets.audio.path` 的相对路径
`../../data/...` 备好 `data/` 目录即可）；`verify.py` 官方口径需 `baselines/<模型>.json`（未随分题材料下发，
缺文件时按设计报错退出、不猜分）。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`（8.0 GB / 24 层） |
| 卡型 | K500SM_AI 64 GB × 1（按赛事口径计 K100-AI），120 SM |
| 节点 | SCNet Notebook，Hygon C86 7185 32C / 503 GB RAM，Ubuntu 24.04 容器 |
| Python | 3.10.21 |
| torch | 2.10.0+das.opt1.dtk2604.2608141812（HIP 6.3.26113） |
| triton | 3.6.0+das185.dtk2604.torch2100.2608141110 |
| vllm | 0.21.0+das.dtk2604.torch2100.2606111143 |
| vllm-hcu | 0.21.0+das.dtk2604.torch2100.2608251643 |
| transformers | 5.5.0 |
| flash_attn | 2.8.3+das185.dtk2604.torch2100.2608142153 |
| 模型路径 | `/root/hygon/models/asr_orig` |
| 端口 | 8000 |

完整原始输出见 [`evidence/env.log`](evidence/env.log)。

> 本实例无容器运行时（无 docker/podman），镜像经 harbor 匿名 pull 后在 Notebook 内解包为 rootfs，
> 驱动 `/opt/hyhal` 由平台注入——这与容器内的实际形态一致（DTK/Python 包来自镜像，驱动来自宿主）。

环境自检（`evidence/env.log` 关键部分原文）：

```text
image: harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10
image_note: 镜像层在 Notebook 内解包到 /root/hygon/rootfs（本实例无容器运行时），DTK/包取自镜像，驱动 /opt/hyhal 由平台注入
dtk: /opt/dtk-26.04  (rocm_path=/root/hygon/rootfs/opt/dtk)
python: 3.10.21
torch: 2.10.0
HIP: 6.3.26113
Device: K500SM_AI x 1
flash_attn                                 2.8.3+das185.dtk2604.torch2100.2608142153.ge8ddf7
transformers                               5.5.0
triton                                     3.6.0+das185.dtk2604.torch2100.2608141110.g65af71
vllm                                       0.21.0+das.dtk2604.torch2100.2606111143.g8c979d
vllm-hcu                                   0.21.0+das.dtk2604.torch2100.2608251643.g135f01
```

超算互联网平台（参赛算力统一使用超算互联网，必填）：

| 项 | 值 |
| --- | --- |
| 区域 | 待控制台补填（共享算力实例内 metadata 服务 100.100.100.200 不可达，读不到 region，不臆造；请在超算互联网控制台「实例详情」确认，如西南一区） |
| Notebook 规格 | Hygon C86 7185 32 核 / 503 GB 内存；overlay 文件系统 1.8 TB（实测可用约 311 GB） |
| 加速卡与显存 | K500SM_AI × 1（按赛事口径计 K100-AI）；64 GB |
| 开发镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`（Notebook 内解包 rootfs 使用） |
| 模型获取方式 | ModelScope（魔搭）模型库下载后 scp 上传实例 `/root/hygon/models/` |
| 服务映射 | 无（后端 `127.0.0.1:8000` 仅实例内访问，未配置平台服务映射） |
| 平台证据 | `platform/`：待补控制台截图（Notebook 创建页、实例详情、模型管理、服务映射）；实例侧自检原文见 `evidence/env.log`（含 hy-smi 输出） |

## 3. 适配优化方案

### 3.1 基础镜像

`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`，对应 `docker_images.md` 中 vLLM 0.21 的推荐镜像。

### 3.2 软件包升级

无。镜像自带 torch / triton / vllm / vllm-hcu / flash_attn / aiter / lmslim 全套 HCU 版本，`requirements-extra.txt` 为空。

### 3.3 代码改动

无模型代码改动。仅两处运行时装配（不涉及模型语义、不影响精度）：

1. `rootfs` 内 `vllm/model_executor/layers/fused_moe/modular_kernel.py` 是指向
   `/usr/local/lib/python3.10/dist-packages/vllm_hcu/...` 的绝对符号链接，解包后指向失效
   （报 `ModuleNotFoundError: vllm.model_executor.layers.fused_moe.modular_kernel`）。
   处理方式：把指向镜像绝对路径的符号链接改指向 `$ROOTFS` 下同一路径，由 `run.sh` 幂等执行。
2. triton 的 HCU 后端从 `ROCM_PATH` 下找 `aillvm` 子目录，宿主 DTK 没有该目录会回退到 `llvm`
   并导致编译出的 `.amdgcn` 缺失（`HSACOError: File operation failed`）。
   处理方式：`ROCM_PATH=$ROOTFS/opt/dtk`，`LD_LIBRARY_PATH`/`PATH` 一并指向镜像内 DTK，驱动仍用宿主 `/opt/hyhal`。

### 3.4 升级步骤

无包升级。复现顺序：解包镜像 → `bash run.sh` → `bash env_check.sh` → `bash smoke.sh` → 精度与性能。

## 4. 服务启动脚本

`env.sh` 里的关键量：`CARD=K100-AI`、`TP=1`、`MAX_LEN=32768`、`MEM_FRAC=0.90`、`PORT=8000`。

最终展开的启动命令（见 `run.sh`，同时写进 `server.log` 首行）：

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/hygon/models/asr_orig \
  --served-model-name qwen3-asr-1.7b \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 8000
```

参数理由：

- `--dtype bfloat16`：官方权重为 bf16，与镜像内 flash-attn 路径匹配；本卡 bf16 与 fp16 实测 matmul
  吞吐相同（见第 7 节前的说明），不额外换精度。
- `--max-model-len 32768`：模型原生上下文；单条音频远小于此，留足余量不影响显存占用（KV cache 按实际长度分配）。
- `--gpu-memory-utilization 0.90`：单卡独占，剩余 10% 留给 DTK 运行时与编译缓存；实测稳态显存 60.2 GB / 64 GB。
- 未使用 `--enforce-eager`：默认走 torch.compile + CUDA Graph，首次启动需一次性编译约 3 分钟，
  之后有 `torch_compile_cache` 复用。
- 未设置 `NCCL_MIN_NCHANNELS` / `NCCL_MAX_NCHANNELS`（单卡也用不到）。

## 5. Smoke 测试

`bash smoke.sh`，原始输出见 [`evidence/smoke.log`](evidence/smoke.log)：

```
== /health
HTTP 200
== /v1/models
{"object":"list","data":[{"id":"qwen3-asr-1.7b", ... "max_model_len":32768 ...
== ASR 中文 (zh.mp3)
{"text":"開放時間：早上九點至下午五點。","usage":{"type":"duration","seconds":6}}
== ASR 英文 (en.mp3)
{"text":"The tribal chief then called for the boy, and presented him with fifty pieces of gold.","usage":{"type":"duration","seconds":8}}
```

curl 复现（读者可直接粘贴）：

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F "file=@zh.mp3" -F "model=qwen3-asr-1.7b"
```

中英两条都是逐字正确的原文转写（`zh.mp3` 为繁体中文字面，模型输出亦为繁体，未改写）。

## 6. 精度测评

**官方分题材料（2026-10-30 下发）实测完成。** 官方 `formal/conditions/qwen3_asr.jsonl`
共 300 条（AISHELL-1 test 100 / LibriSpeech test-clean 100 / test-other 100），
seed=42、temperature=0、bf16，全部在 K500 上跑通：**300/300 成功、零失败**。

打分口径：中文按字级编辑距离（CER）、英文按词级编辑距离（WER），均去除标点后计算
（脚本 `verify_formal.py`，逐条原始输出见 `evidence/formal_asr/outputs/`，
汇总见 `evidence/formal_asr/verify_result.json`，官方反馈表 300 行见
`evidence/formal_asr/反馈表-qwen3_asr.csv`）。

| 测试集 | 条数 | 官方 GPU 参考 | 本机 K500 实测 |
| --- | --- | --- | --- |
| AISHELL-1 test CER | 100 | — | **1.00%** |
| LibriSpeech test-clean WER | 100 | 1.63 | **1.11%** |
| LibriSpeech test-other WER | 100 | 3.38 | **3.31%** |

LibriSpeech 两个子集与官方 README 公布的 GPU 参考值（clean 1.63 / other 3.38）一致，
判定模型精度在 K500（DTK 26.04 + vLLM 0.21.0）上无损。

执行细节（如实说明）：官方 `baselines/<模型>.json` 基线文件未随分题材料下发，本节数字由
官方条件清单直接打分产出，非对照 verify.py 基线容差；vLLM `/v1/audio/transcriptions`
端点在本环境对 PCM wav/flac 直读会产生重复/空输出（详见第 8 节），故全部音频先经
ffmpeg 转 16 kHz 单声道 mp3 后提交——mp3 通路经同一内容对照验证输出正确，模型与权重未改动。

`verify_result.json` 原文（`evidence/formal_asr/verify_result.json`，`verify_formal.py` 重跑可逐字节复现）：

```json
{
 "missing": [],
 "AISHELL-1 CER": {
  "errors": 14,
  "tokens": 1403,
  "utterances": 100,
  "rate": 1.0
 },
 "LibriSpeech WER": {
  "errors": 78,
  "tokens": 3917,
  "utterances": 200,
  "rate": 1.99
 },
 "LibriSpeech test-clean WER": {
  "errors": 26,
  "tokens": 2346,
  "utterances": 100,
  "rate": 1.11
 },
 "LibriSpeech test-other WER": {
  "errors": 52,
  "tokens": 1571,
  "utterances": 100,
  "rate": 3.31
 }
}
```

格式说明：T04 的评测器、数据集与判定口径随分题材料下发（模板允许表格列按实际指标调整），
故本件 `verify_result.json` 为 CER/WER 打分原始输出（`rate` 为百分数），与 T01 的
detail`{got/ref/worse/tol}` 基线容差格式不同；`missing: []` 表示 300 条全部有转写输出。

## 7. 性能参考

官方写明「生成类与语音类的测法随分题材料下发」，T04 压测脚本未下发，因此这里给出**自定但可复现**的
ASR 口径（RTF = 墙钟时间 / 音频时长，越小越好），并明确标注与官方口径不同。原始输出见
[`evidence/bench.txt`](evidence/bench.txt)。

测试集：3 条公开样例音频（6 s 中文、8 s 英文、6 s 其他），按并发要求重复提交，全部 HTTP 200 且返回非空转写。

| 并发 | 请求数 | 成功 | 墙钟 | 音频总长 | RTF | 吞吐 | QPS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 63 | 63 | 19.12 s | 420 s | 0.046× | 21.97 audio-s/s | 3.30 |
| 8 | 48 | 48 | 2.68 s | 320 s | 0.008× | 119.53 audio-s/s | 17.93 |
| 32 | 96 | 96 | 3.07 s | 640 s | 0.005× | 208.23 audio-s/s | 31.23 |

单并发 22 倍实时；32 并发下 208 倍实时（约 31 条/秒），随并发近似线性到 8、之后趋于饱和。

另附硬件基线（同一实例、同一 DTK，非本模型）：8192³ matmul fp16/bf16 均 66.5 TFLOPS，显存 copy 670 GB/s。
bf16 与 fp16 数字完全一致，怀疑该版本 bf16 走了 fp16 通路，未进一步验证，仅作参考。

## 8. 注意事项

- **`-hf` 权重不能用**：`Qwen/Qwen3-ASR-1.7B-hf` 的 707 个张量全部带 `model.` 前缀，vLLM 0.21 的
  `Qwen3ASRForConditionalGeneration` 不剥该前缀，加载直接报
  `ValueError: There is no module or parameter named 'model'`。请用 `Qwen/Qwen3-ASR-1.7B`（`thinker.` 前缀）。
- **必须指向镜像内 DTK**：`ROCM_PATH` 若停在宿主 `/opt/dtk`，triton 找不到 `aillvm`，退到 `llvm` 后
  kernel 编译出的 `.amdgcn` 缺失，报 `HSACOError`——现象是 EngineCore 启动失败，容易误判成模型问题。
- **解包镜像要修符号链接**：见 3.3。
- **首次启动约 3 分钟**是 torch.compile 编译，不是卡住；`torch_compile_cache` 命中后会快很多。
- aiter 会提示 `NUMA balancing is enabled, which may cause errors`，本实例无权限写
  `/proc/sys/kernel/numa_balancing`，实测未影响结果，保留原样。
- 纯文本请求（不带音频）能返回，但输出无意义（模型在无音频时输出 `<asr_text>` 类空结果），
  不能当作文本 LLM 使用，也不要用它做语言侧压测。
- `/v1/chat/completions` 传本地文件路径需要 `--allowed-local-media-path`，本方案未开启；
  转写请用 `/v1/audio/transcriptions`。
- **PCM wav/flac 直读输出异常（仅音频解码路径）**：同一音频内容分别以 wav 与 mp3 送
  `/v1/audio/transcriptions`，wav 返回重复单字/空文本、mp3 返回正确转写——判定服务端 PCM
  解码路径问题。官方 300 条跑批因此统一先 `ffmpeg` 转 16 kHz 单声道 mp3 再提交
  （见 `evidence/formal_asr/run_asr.py`），mp3 通路输出经与参考文本对照验证正确，
  模型与权重未改动。
- **打包布局说明**：`run_asr.py` / `verify_formal.py` 的路径常量已对齐本仓库布局
  （条件清单 `formal/conditions/qwen3_asr.jsonl`、逐条输出 `evidence/formal_asr/outputs/`、
  打分结果 `evidence/formal_asr/verify_result.json`），归一化与编辑距离判分算法未改，
  本地重跑 `verify_formal.py` 与归档 `verify_result.json` 逐字节一致。300 条音频资产
  未随仓库分发（分题材料原件，约数 GB），按条件清单相对路径备好 `data/` 后即可复现批跑。

---

### 仓库内容

| 文件 | 对应报告章节 | 说明 |
| --- | --- | --- |
| `env.sh` | 2 / 4 | 唯一需要填的变量表 |
| `run.sh` | 4 | 装配环境 + 起服务 + 轮询 /health |
| `env_check.sh` | 2 | 采集版本 → `evidence/env.log` |
| `smoke.sh` | 5 | 健康检查 + 中英转写 → `evidence/smoke.log` |
| `verify.py` | 6 | 官方判分逻辑（`--selfcheck` 可用；缺基线文件时不臆造分数） |
| `verify_formal.py` | 6 | 官方分题 300 条 CER/WER 打分 → `evidence/formal_asr/verify_result.json` |
| `formal/conditions/` | 6 | 官方分题条件清单 `qwen3_asr.jsonl`（2026-10-30 下发原件） |
| `bench.sh` / `asr_bench.py` | 7 | ASR 并发吞吐 → `evidence/bench.txt` |
| `evidence/` | — | 上述原始产物（`formal_asr/` 含 300 条逐条输出与官方反馈表） |
