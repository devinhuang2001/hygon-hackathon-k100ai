# Tencent-Hunyuan/AuK on K100-AI（K500SM_AI 64GB）+ 官方源码推理

> 本报告按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> 卡型说明：实测实例为 `K500SM_AI`，按赛事口径计入 `K100-AI`。

## 1. 简介

本文档记录 `Tencent-Hunyuan/AuK`（1.5B 语音生成/编辑基础模型）在 `K100-AI（K500SM_AI 64GB）`（精度 `bf16`）环境的适配与验证结果。

AuK 是腾讯混元开源的语音生成与编辑基础模型（扩散 Transformer + 层融合权重，`auk_base.safetensors` + `vae.safetensors`，共 1.69B 参数），MLLM 编码器为 `Qwen/Qwen2.5-Omni-3B`。本次执行官方分题 `auk.jsonl` 的 120 条零样本 TTS（voice clone）：Seed-TTS eval zh/en 各 60 条（40 组 × seed 42/2026/1234），按清单参数（nfe=32、cfg_strength=2.0、sway_sampling_coef=-1.0）生成 24 kHz 单声道 wav。

与官方部署方式的核心差异：无模型代码改动，使用 GitHub `Tencent-Hunyuan/AuK` 官方源码直调 `AukInfer`；运维层面在离线环境补齐依赖链（torchdiffeq / x_transformers / einx / torch_einops_utils / librosa 等），并以 ffmpeg 等价替代 torchaudio 2.10 的 load/save（其依赖的 TorchCodec 在镜像内不可用，替代不影响模型计算）。

相关获取地址：

- AuK 权重：https://github.com/Tencent-Hunyuan/AuK（release `auk_base.safetensors` / `vae.safetensors`）
- 编码器权重：https://www.modelscope.cn/models/Qwen/Qwen2.5-Omni-3B
- 提交仓库：https://github.com/devinhuang2001/hygon-hackathon-k100ai/tree/main/04-AuK
- 光合社区代码托管（赛事要求）：https://developer.sourcefind.cn/codes/g_x8d7w4tr/AuK-K100-AI

参考文档：

- AuK 官方源码与 README：https://github.com/Tencent-Hunyuan/AuK
- MLLM 编码器 Qwen2.5-Omni-3B：https://www.modelscope.cn/models/Qwen/Qwen2.5-Omni-3B

一键复现（T04 交付物按分题材料为「反馈表 + 环境说明 + 逐条 outputs」，故不设 env_check.sh / smoke.sh / verify.py / bench.sh）：

```bash
git clone https://github.com/devinhuang2001/hygon-hackathon-k100ai && cd hygon-hackathon-k100ai/04-AuK
bash run.sh          # 装配依赖 + 批跑官方 120 条 → evidence/formal_auk/outputs/
```

前置（实例无外网，见 3.2/3.3）：预先按 manylinux2014_x86_64 + py3.10 拉 wheel 上传并
`pip3 install --no-index --find-links` 安装；AuK 源码按 3.3 以 PYTHONPATH 引入。
官方条件清单 `formal/auk.jsonl`（2026-10-30 下发原件）已在仓库内。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 卡型 | K500SM_AI 64 GB × 1（按赛事口径计 K100-AI），120 SM |
| 节点 | SCNet Notebook，Hygon C86 7185 32C / 503 GB RAM，Ubuntu 24.04 容器 |
| 镜像基座 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`（rootfs 解包，Python/torch 同镜像） |
| Python | 3.10.21 |
| torch | 2.10.0+das.opt1.dtk2604.2608141812（HIP 6.3.26113） |
| transformers | 5.5.0（镜像自带） |
| AuK 源码 | GitHub Tencent-Hunyuan/AuK 官方源码（未改动），`src/` 进 PYTHONPATH |
| 模型路径 | AuK: `/root/hygon/models/auk`；编码器: `/root/hygon/models/qwen-omni` |
| 推理方式 | `auk.infer.infer_auk.AukInfer` 直调（in-process），无 HTTP 服务 |

环境自检（本件环境原文见 [`evidence/formal_auk/环境说明.txt`](evidence/formal_auk/环境说明.txt)，关键部分摘录）：

```text
设备信息：
- Hygon K500SM_AI 64GB × 1（按赛事口径计 K100-AI），DTK 26.04，SCNet Notebook 容器

框架及版本：
- vLLM 0.21.0+das.dtk2604.torch2100.2606111143（vllm-hcu 0.21.0+das.dtk2604.2608251643），
  镜像 harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10（解包 rootfs 方式运行）
- AuK 使用 GitHub Tencent-Hunyuan/AuK 官方源码（未改动），transformers 5.5.0，torch 2.10.0+das.opt1.dtk2604
```

本件与 01/03 件为同一 K500 实例、同一镜像 rootfs；实例级 `env.log` 原文见 01 件
`evidence/env.log`（python 3.10.21 / torch 2.10.0 / HIP 6.3.26113 / Device: K500SM_AI x 1）。

超算互联网平台（参赛算力统一使用超算互联网，必填）：

| 项 | 值 |
| --- | --- |
| 区域 | 待控制台补填（共享算力实例内 metadata 服务 100.100.100.200 不可达，读不到 region，不臆造；请在超算互联网控制台「实例详情」确认，如西南一区） |
| Notebook 规格 | Hygon C86 7185 32 核 / 503 GB 内存；overlay 文件系统 1.8 TB（实测可用约 311 GB） |
| 加速卡与显存 | K500SM_AI × 1（按赛事口径计 K100-AI）；64 GB |
| 开发镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`（Notebook 内解包 rootfs 使用） |
| 模型获取方式 | AuK 权重取 GitHub 官方 release、编码器 Qwen2.5-Omni-3B 取 ModelScope，均 scp 上传实例 `/root/hygon/models/` |
| 服务映射 | 无（in-process 批跑，无 HTTP 服务） |
| 平台证据 | `evidence/platform/`：01-env-selfcheck.png（环境自检：K500SM_AI/68.7GB/gfx928 + torch 2.10.0+das + DTK 26.04 + pip 清单 + 120 wav ffprobe 抽检 + batch log 全 OK）；02-outputs-verify.png（产物验证）；03-model-library.png（AI 社区模型管理/模型库页）；04-image-library.png（开发镜像库页）；05-model-auk-search.png（AuK 模型检索）。Notebook 创建页/服务映射为控制台私有页，实例侧等效证据见 `evidence/env.log` 与 `evidence/formal_auk/环境说明.txt` |

## 3. 适配优化方案

### 3.1 基础镜像

同 01/03 件（vLLM 0.21 镜像 rootfs），Python/torch/DTK 全部复用。

### 3.2 软件包升级

镜像基础上离线安装（`requirements-extra.txt` 口径，均为纯 Python 包，不动 torch/transformers）：

| 包 | 版本 | 用途 |
| --- | --- | --- |
| torchdiffeq | 0.2.5 | AuK 扩散采样 ODE |
| x_transformers | 0.42.0 | AuK 模型依赖 |
| einx / multimethod 1.12 / frozendict | — | x_transformers 依赖链 |
| torch_einops_utils / einops | 0.2.0 / 0.8.2 | AuK 注意力实现 |
| audioread / librosa 0.11.0（含 numba/llvmlite/soxr/soundfile 等） | — | qwen_omni_utils 音频处理 |

安装方式：本地 `pip download --platform manylinux2014_x86_64 --python-version 3.10` 拉 wheel → scp 上传 → `pip3 install --no-index --find-links`。实例无外网，全程离线。

### 3.3 代码改动

无模型代码改动。三处运行适配（不影响模型计算与输出）：

1. **torchaudio load/save 替代**：镜像内 torchaudio 2.10 的 `load`/`save` 需要 TorchCodec（未随镜像提供）。
   `run_auk.py` 以 ffmpeg（f32le 16k/24k PCM）等价替代读写，张量形状与官方 `_load_audio` 约定一致
   （返回 `[1, T]` 二维；官方在字符串分支会 `mean(dim=0)`，一维输入会塌成单采样点——已规避）。
2. **qwen-omni 编码器目录补齐**：远端目录原有 3 个权重分片但缺 `config.json` / `tokenizer*` /
   `model.safetensors.index.json`，从 ModelScope 下载小文件补齐（权重分片本身未变）。
3. **AuK 源码以 PYTHONPATH 方式引入**（`src/` 不 pip install -e，避免改镜像 site-packages）。

### 3.4 升级步骤

解包镜像 → 离线装 3.2 列出的 wheel → 上传 AuK 源码与权重 → `bash run.sh`（批跑 120 条）。

## 4. 服务启动脚本

AuK 无 HTTP 服务，走 in-process 批跑。`run_auk.py` 关键调用：

```python
engine = AukInfer(
    auk_path="/root/hygon/models/auk",
    qwen_path="/root/hygon/models/qwen-omni",
    cpu_offload=False)
engine.generate(
    audio=ref_wav,          # 参考音频（ffmpeg 读入 [1,T], 24k）
    instruction=...,        # 按清单 instruction/text
    gen_seconds=p["gen_seconds"], nfe=p["nfe"],
    cfg_strength=p["cfg_strength"], sway_sampling_coef=p["sway_sampling_coef"],
    seed=r["seed"])
```

参数全部取自官方条件清单（nfe=32、cfg_strength=2.0、sway_sampling_coef=-1.0、gen_seconds 与 seed 按条目），未做任何改动。

## 5. Smoke 测试

单条零样本 TTS（zh 参考音频 + 中文目标文本），输出 24 kHz 单声道 wav，时长落在清单 `expected.duration_min_s/max_s` 区间（6.02–6.12 s）。命令：

```bash
python3 run_auk.py          # 全量 120 条，逐条 [auk] OK <id>
```

## 6. 精度测评

**官方分题材料（2026-10-30 下发）实测完成。** 官方 `formal/conditions/auk.jsonl` 共 120 条
（Seed-TTS eval zh 60 / en 60，seed 42/2026/1234），全部在 K500 上跑通：**120/120 成功、零失败**。
逐条原始输出见 [`evidence/formal_auk/outputs/`](evidence/formal_auk/outputs/)，
官方反馈表 120 行见 [`evidence/formal_auk/反馈表-auk.csv`](evidence/formal_auk/反馈表-auk.csv)。

输出规格（逐条验证）：24 kHz 单声道 wav，时长全部落在清单 `expected` 的 `duration_min_s`/`duration_max_s` 区间内。

**关于音质指标（SIM / WER / MOS）：** 官方清单不附带评分脚本与参考打分（`expected.kind=audio` 仅约束
格式与时长），本次按官方口径**只交原始输出、不自造分数**。克隆相似度等指标需官方评测方统一计算。

批次汇总（按真实跑批清点，非官方评分——无分数类 verify_result.json）：

```json
{
  "MODEL_ID": "Tencent-Hunyuan/AuK",
  "CARD": "K100-AI (K500SM_AI 64GB)",
  "FRAMEWORK": "AuK official source (AukInfer, in-process)",
  "PRECISION": "bf16",
  "batch": {
    "conditions": "formal/auk.jsonl",
    "total": 120,
    "succeeded": 120,
    "failed": 0,
    "output": "24kHz mono wav, duration within expected.duration_min_s/max_s"
  },
  "metric": {"name": "SIM/WER/MOS", "value": null, "status": "待官方评测方统一计算"}
}
```

反馈表（`evidence/formal_auk/反馈表-auk.csv`，120 行）列名：
`condition_id,model,status,output_path,error,parameter_changes`。

## 7. 性能参考

官方压测脚本未下发，给出**自定但可复现**的批跑口径（日志见 `run_auk.py` 执行输出）：

| 项 | 值 |
| --- | --- |
| 任务 | 官方 120 条（zh/en 各 60，gen_seconds ≈ 4–11 s/条） |
| 单条延迟 | ≈ 8 s（含 VAE 解码；扩散 32 步 + 扩散采样） |
| 总墙钟 | ≈ 16 分钟（120 条，含引擎/权重加载约 3 分钟） |
| 显存 | 实测峰值 ≈ 25 GiB（bf16，与官方 A800 实测 24.8–25.0 GiB 一致） |

## 8. 注意事项

- **TorchCodec 缺失**：torchaudio 2.10 的 `load`/`save` 在本镜像不可用，已用 ffmpeg 等价替代
  （见 3.3）。注意 load 返回 `[1, T]` 二维张量——AuK 官方 `_load_audio` 在字符串分支有
  `mean(dim=0)`，一维输入会塌成单采样点导致 conv1d 报错。
- **依赖全离线**：实例无外网，3.2 的 wheel 需预先下载上传。librosa 0.11 链（numba/llvmlite）
  按 manylinux2014_x86_64 + py3.10 下载，Windows wheel 不适用。
- **qwen-omni 缺配置文件**：只有权重分片不够，缺 `config.json`/`tokenizer*`/`model.safetensors.index.json`
  时 AukInfer 加载报 KeyError，补齐小文件即可（权重不动）。
- **seed 语义**：清单 seed=42/2026/1234 决定扩散初始噪声，同一参考音频 + 同 seed 可复现。
- 输出时长由 `gen_seconds` 控制（清单逐条给出），`duration_budget_rule` 只是校验口径。

---

### 仓库内容

| 文件 | 对应报告章节 | 说明 |
| --- | --- | --- |
| `env.sh` | 2 / 4 | 变量表（AuK/编码器路径、PYTHONPATH） |
| `run.sh` | 4 | 装配环境 + 批跑 120 条 |
| `run_auk.py` | 3 / 4 | 批跑脚本（含 ffmpeg load/save 替代） |
| `formal/auk.jsonl` | 6 | 官方分题条件清单（2026-10-30 下发原件） |
| `evidence/` | — | 原始产物（formal_auk/ 含 120 条 wav 与官方反馈表） |
