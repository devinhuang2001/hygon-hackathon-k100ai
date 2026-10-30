# Qwen3-ForcedAligner-0.6B on K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0

> 本报告按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> 卡型说明：实测实例为 `K500SM_AI`，按赛事口径计入 `K100-AI`。

## 1. 简介

本文档记录 `Qwen/Qwen3-ForcedAligner-0.6B` 在 `K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0`（精度 `bf16`）环境的适配与验证结果。

Qwen3-ForcedAligner-0.6B 是 Qwen3-ASR 系列配套的强制对齐模型：输入音频与对应文本，以非自回归方式输出词级时间戳，支持 11 种语言、最长 5 分钟语音。结构同为 `Qwen3ASRForConditionalGeneration`（thinker 类型 `qwen3_forced_aligner`，`classify_num=5000`），合计 0.92B 参数，BF16 单文件约 1.8 GB。官方 README 报告时间戳偏移指标 AAS 在 MFA 标注集上均值 42.9 ms。

与官方部署方式的核心差异：官方经 `qwen-asr` 包以 Transformers 方式加载；本方案走 **vLLM 0.21 的 in-process pooling（`runner="pooling"` + `pooling_task="token_classify"`）** 路径，对齐算法（分词 → `<timestamp>` 插入 → 时间格分类 → LIS 平滑）逐行复刻官方 `qwen_asr/inference/qwen3_forced_aligner.py`。未升级任何包、未改动一行模型代码；运维层面复用同实例已验证的两处装配（rootfs 符号链接改指向、`ROCM_PATH` 指向镜像内 DTK）。

相关获取地址：

- 权重下载：https://www.modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B
- 镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`
- 提交仓库：https://github.com/devinhuang2001/hygon-hackathon-k100ai/tree/main/03-Qwen3-ForcedAligner-0.6B
- 光合社区代码托管（赛事要求）：https://developer.sourcefind.cn/codes/g_x8d7w4tr/Qwen3-ForcedAligner-0.6B-K100-AI

参考文档：

- Qwen3-ForcedAligner-0.6B 官方模型卡：https://www.modelscope.cn/models/Qwen/Qwen3-ForcedAligner-0.6B
- 对齐算法出处：官方 `qwen_asr/inference/qwen3_forced_aligner.py`（Qwen3-ASR 发布物，本仓库 `fa_align.py` 逐行复刻）
- vLLM pooling（token classification）文档：https://docs.vllm.ai

一键复现（T04 交付物按分题材料为「反馈表 + 逐条 outputs」，故不设 env_check.sh / smoke.sh / verify.py / bench.sh）：

```bash
git clone https://github.com/devinhuang2001/hygon-hackathon-k100ai && cd hygon-hackathon-k100ai/03-Qwen3-ForcedAligner-0.6B
bash run.sh                                              # relink 装配 + 批跑官方 30 条 → evidence/formal_fa/outputs/
python3 fa_align.py <audio.wav> "<文本>" <language>        # 单条对齐（第 5 节 smoke 同款）
```

官方条件清单 `formal/forced_aligner.jsonl`（2026-10-30 下发原件）已在仓库内，`run.sh` 即
模板验证顺序中的「装配 + 评测」一步；反馈表与逐条输出见 `evidence/formal_fa/`。

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
| 模型路径 | `/root/hygon/models/fa06` |
| 推理方式 | vLLM in-process（`LLM(runner="pooling")`），无 HTTP 服务 |

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
| 服务映射 | 无（in-process 批跑，无 HTTP 服务） |
| 平台证据 | `platform/`：待补控制台截图（Notebook 创建页、实例详情、模型管理、服务映射）；实例侧自检原文见 `evidence/env.log`（含 hy-smi 输出） |

## 3. 适配优化方案

### 3.1 基础镜像

`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`，对应 `docker_images.md` 中 vLLM 0.21 的推荐镜像。

### 3.2 软件包升级

无。镜像自带 torch / triton / vllm / vllm-hcu / flash_attn / aiter 全套 HCU 版本，`requirements-extra.txt` 为空。

### 3.3 代码改动

无模型代码改动。对齐算法（`fa_align.py`）按官方 `qwen_asr/inference/qwen3_forced_aligner.py` 复刻：

1. 文本按官方 `tokenize_space_lang` 语义切词（空格切分、CJK 逐字），
   以 `"<timestamp><timestamp>".join(words) + "<timestamp><timestamp>"` 插入时间标记，
   前置 `"<|audio_start|><|audio_pad|><|audio_end|>"` 占位音频；
2. 模型对每个 `<timestamp>` 位置分类出时间格索引，`timestamp_ms = idx * 80`（官方 `timestamp_segment_time`）；
3. 用最长递增子序列（官方 `fix_timestamp` 同构实现）把跳变格拉回单调，再按 `(start, end)` 成对切给每个词。

仅两处运行时装配（与 01 件同实例、同一处理，不涉及模型语义、不影响精度）：

1. `rootfs` 内 `vllm/.../modular_kernel.py` 绝对符号链接解包后失效，改指向 `$ROOTFS` 下同一路径；
2. triton 的 HCU 后端从 `ROCM_PATH` 找 `aillvm`，设 `ROCM_PATH=$ROOTFS/opt/dtk`。

### 3.4 升级步骤

无包升级。复现顺序：解包镜像 → `bash run.sh`（装配 + 批跑 30 条）→ 查看 `evidence/formal_fa/outputs/`。

## 4. 服务启动脚本

FA 无 HTTP 服务，走 in-process 批跑。`env.sh` 里的关键量：`MODEL=/root/hygon/models/fa06`、`MAX_LEN=8192`、`MEM_FRAC=0.35`。

`run_fa.py` 展开的加载参数（见 `evidence/fa_full.log` 首次跑通日志中的 non-default args）：

```python
LLM(
    model="/root/hygon/models/fa06",
    runner="pooling",
    hf_overrides={"architectures": ["Qwen3ASRForcedAlignerForTokenClassification"]},
    dtype="bfloat16",
    max_model_len=8192,
    enforce_eager=True,
    gpu_memory_utilization=0.35,
    trust_remote_code=True,
)
```

参数理由：

- `runner="pooling"` + `pooling_task="token_classify"`：官方模型是 token 分类头（`classify_num=5000`），
  不走自回归生成，vLLM pooling 路径直接给出每个位置的类别 logits。
- `hf_overrides` 覆盖 architectures：官方 config 的 thinker 类型为 `qwen3_forced_aligner`，
  vLLM 0.21 需显式指定 `Qwen3ASRForcedAlignerForTokenClassification` 才能路由到分类实现。
- `max_model_len=8192`：对齐输入为音频占位 + 文本 token，单条远小于此。
- `enforce_eager=True`：批量对齐一次性任务，省去 torch.compile 编译等待；分类头单次前向，图捕获收益小。
- `gpu_memory_utilization=0.35`：0.6B 模型显存占用小，为同卡其他任务留余量。

音频统一经 ffmpeg 重采样为 16 kHz 单声道 f32 后以 `(np.ndarray, 16000)` 元组传入引擎。

## 5. Smoke 测试

单条对齐（`fa_align.py` 直调），原始输出见 [`evidence/fa_full.log`](evidence/fa_full.log)：

```json
{
 "items": [
  {"text": "The",     "start": 2.96, "end": 3.04},
  {"text": "tribal",  "start": 3.12, "end": 3.12},
  {"text": "chief",   "start": 3.36, "end": 3.44},
  ...
 ]
}
```

输入为 7.18 s 英文音频 + 对应文本，16 个词全部得到起止时间（单位秒），时间戳随音频推进单调。命令：

```bash
python3 fa_align.py audio.wav "The tribal chief then called for the boy..." English
```

## 6. 精度测评

**官方分题材料（2026-10-30 下发）实测完成。** 官方 `formal/conditions/forced_aligner.jsonl`
共 30 条（LibriSpeech test-clean），seed=42、bf16，全部在 K500 上跑通：**30/30 成功、零失败**。
逐条原始输出（词级起止时间 JSON，单位秒）见 [`evidence/formal_fa/outputs/`](evidence/formal_fa/outputs/)，
官方反馈表 30 行见 [`evidence/formal_fa/反馈表-forced_aligner.csv`](evidence/formal_fa/反馈表-forced_aligner.csv)。

**关于 AAS（时间戳偏移）指标：** 官方条件清单标注 `BOUNDARIES_PENDING`——本次下发材料只收集预测、
不提供参考边界，故按官方口径**只返回预测、不计算边界准确率**。官方 README 公布的 42.9 ms AAS
来自 MFA 标注集对照，本次无对应参考数据，按赛事红线不臆造分数。待参考边界下发后可直接用
本仓库的 `evidence/formal_fa/outputs/` 对照计算。

输出格式（逐条 outputs 实际结构）：每条一个 JSON，含 `condition_id` / `language` / `unit`，
词级起止时间在 `words[]`（每个词 `text` / `start` / `end`，单位秒）。

批次汇总（按真实跑批清点，非官方评分——清单标注 `BOUNDARIES_PENDING`，无分数类 verify_result.json）：

```json
{
  "MODEL_ID": "Qwen/Qwen3-ForcedAligner-0.6B",
  "CARD": "K100-AI (K500SM_AI 64GB)",
  "FRAMEWORK": "vLLM 0.21.0 in-process pooling (token_classify)",
  "PRECISION": "bf16",
  "batch": {
    "conditions": "formal/forced_aligner.jsonl",
    "total": 30,
    "succeeded": 30,
    "failed": 0
  },
  "metric": {"name": "AAS", "value": null, "status": "BOUNDARIES_PENDING"}
}
```

反馈表（`evidence/formal_fa/反馈表-forced_aligner.csv`，30 行）列名：
`condition_id,model,status,output_path,error,parameter_changes`。
单条输出原文节选（逐字取自 `evidence/formal_fa/outputs/1089-134686-0000.json`，全文 31 词）：

```json
{
 "condition_id": "1089-134686-0000",
 "language": "English",
 "unit": "seconds",
 "words": [
  {"text": "HE", "start": 0.48, "end": 0.72},
  {"text": "HOPED", "start": 0.72, "end": 1.04},
  {"text": "THERE", "start": 1.04, "end": 1.12},
  ...
  {"text": "SAUCE", "start": 9.44, "end": 10.08}
 ]
}
```

## 7. 性能参考

官方写明「生成类与语音类的测法随分题材料下发」，T04 压测脚本未下发，因此这里给出**自定但可复现**的
批跑口径（原始日志见 [`evidence/fa_batch.log`](evidence/fa_batch.log)）：

| 项 | 值 |
| --- | --- |
| 任务 | 官方 30 条（LibriSpeech test-clean，8–18 词/条） |
| 总墙钟 | 105 s（17:50:25 → 17:52:10） |
| 含引擎加载 | 是（LLM 初始化 + 权重加载约 35 s，与批跑在同一进程） |
| 对齐吞吐 | 30 条 / 约 70 s 推理 ≈ 0.43 条/s（模型加载后） |

单条延迟约 2 s（音频 5–15 s），非自回归分类头单次前向，延迟随音频长度近似线性。
0.6B 模型加载仅占显存 0.35 上限，同卡可与其他任务共存。

## 8. 注意事项

- **算法复刻范围**：`fa_align.py` 复刻官方 `qwen_asr/inference/qwen3_forced_aligner.py` 的
  分词、`<timestamp>` 插入、`fix_timestamp`（LIS 平滑）与成对切分逻辑；时间格宽度 80 ms
  对应官方 `timestamp_segment_time=80`。`<timestamp>` token id 151705 取自官方 tokenizer。
- **BOUNDARIES_PENDING**：见第 6 节——无参考边界时只交预测，不算 AAS。
- **解包镜像要修符号链接**、**`ROCM_PATH` 必须指向镜像内 DTK**：同 01 件（K500 实例环境共性）。
- **`-hf` 权重注意事项**：与 01 件同理，vLLM 加载需 `thinker.` 前缀的原版权重目录；
  如遇 `Qwen3-ForcedAligner-0.6B-hf` 加载报参数名不匹配，请换用 ModelScope 原版目录。
- 纯文本对齐（不给音频）无意义——音频占位 token 展开后参与分类，无音频时时间格输出退化。
- 输出中可能出现 `start == end` 的词（如 smoke 样例中的 "tribal"），这是分类格粒度 80 ms 下
  两格相同所致，属模型正常行为，与官方输出一致。

---

### 仓库内容

| 文件 | 对应报告章节 | 说明 |
| --- | --- | --- |
| `env.sh` | 2 / 4 | 变量表（模型路径、显存上限等） |
| `run.sh` | 4 | 装配环境 + relink + 批跑 30 条 |
| `fa_align.py` | 3 / 4 | 对齐核心算法（复刻官方，单条 CLI 可用） |
| `run_fa.py` | 4 / 6 | 官方条件清单 30 条批跑 → `evidence/formal_fa/outputs/` |
| `formal/forced_aligner.jsonl` | 6 | 官方分题条件清单（2026-10-30 下发原件） |
| `evidence/` | — | 原始产物（env.log / fa_full.log / fa_batch.log / formal_fa/ 含 30 条逐条输出与官方反馈表） |
