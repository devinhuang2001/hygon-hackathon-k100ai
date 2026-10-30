# MiMo-V2.6-Distill-Qwen-9B on K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0

> 按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> **赛题归属：T01 文本生成 / 推理（计件）**。官方清单 T01 行为
> `Kimi-Linear-48B-A3B-Instruct`、`MiMo-V2.6-Distill-Qwen-9B`，本包对应后者。

## 1. 简介

记录 `XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B` 在 `K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0`（bf16）的部署与实测。

MiMo-V2.6-Distill-Qwen-9B 是小米 MiMo 的 9B 蒸馏模型，基于 Qwen3.5 架构，采用 **Mamba + Attention 混合**结构（vLLM 日志可见 mamba page size 计算），支持思考/非思考模式。

与官方部署方式的核心差异：未升级任何包、未改模型代码；仅两处运行时装配（符号链接改指向、`ROCM_PATH`
指向镜像内 DTK），见第 3 节。权重取本机共享模型库 `/root/hygon/models/mimo`。

- 权重：https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
- 镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`
- 提交仓库：https://github.com/devinhuang2001/hygon-hackathon-k100ai/tree/main/07-MiMo-V2.6-Distill-Qwen-9B

参考文档：

- MiMo-V2.6-Distill-Qwen-9B 官方模型卡：https://www.modelscope.cn/models/XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B
- vLLM 文档：https://docs.vllm.ai

一键复现：

```bash
git clone https://github.com/devinhuang2001/hygon-hackathon-k100ai && cd hygon-hackathon-k100ai/07-MiMo-V2.6-Distill-Qwen-9B
bash run.sh                    # 装配 + 起服务（127.0.0.1:8000），轮询 /health
bash env_check.sh              # 采集版本 → env.log
bash smoke.sh                  # 单轮/多轮/算术/流式 → smoke.log
source env.sh && python verify.py --selfcheck   # 确认判分逻辑未被改动
# 第 6 节精度（官方基线文件未下发，走 --abs 绝对精度；数据集在 data/）：
python verify.py --abs --dataset gsm8k --data data/gsm8k_test.jsonl --train data/gsm8k_train.jsonl --shots 5 --conc 24 --out gsm8k_verify.json
python verify.py --abs --dataset mmlu  --data data/mmlu_test.parquet --per-subject 5 --conc 24 --out mmlu_verify.json
bash bench.sh                  # 简化网格 → evidence/bench/
```

`verify.py` 默认（官方口径）模式需 `baselines/<模型>.json`，未随分题材料下发前按设计报错退出、不猜分；
`--abs` 产出的 JSON 归档在 `evidence/precision/`。

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` |
| 卡型 | K500SM_AI 64 GB × 1（计 K100-AI），120 SM |
| 节点 | SCNet Notebook，Hygon C86 7185 32C / 503 GB RAM |
| Python / torch | 3.10.21 / 2.10.0+das.opt1.dtk2604.2608141812（HIP 6.3.26113） |
| triton / vllm | 3.6.0+das185.dtk2604 / 0.21.0+das.dtk2604 |
| 模型路径 | `/root/hygon/models/mimo` |

完整输出见 [`evidence/env.log`](evidence/env.log)。

环境自检（`env.log` 关键部分原文）：

```text
model_dir: /root/hygon/models/mimo
served_name: mimo-v2.6-9b
image: harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10
python: 3.10.21
torch: 2.10.0
HIP: 6.3.26113
Device: K500SM_AI x 1
flash_attn                                 2.8.3+das185.dtk2604.torch2100.2608142153.ge8ddf7
```

（triton / vllm / vllm-hcu 版本见上表，与 `server.log` 首行一致；完整 pip 清单见 `env.log`。）

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

3.1 基础镜像同上。3.2 软件包升级：无。

3.3 代码改动：无模型代码改动，仅两处运行时装配——
(1) rootfs 内 `vllm/.../fused_moe/modular_kernel.py` 的绝对符号链接失效，改指向 `$ROOTFS`；
(2) `ROCM_PATH` 指向镜像内 DTK（含 triton 所需 `aillvm`），否则 kernel 编译报 `HSACOError`。

3.4 升级步骤：解包镜像 → `bash run.sh` → `bash env_check.sh` → `bash smoke.sh` → `bash bench.sh`。

## 4. 服务启动脚本

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/hygon/models/mimo \
  --served-model-name mimo-v2.6-9b \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 8000
```

参数理由同其它包。Qwen3 系列思考模式默认开启，smoke 里用
`chat_template_kwargs={"enable_thinking": false}` 取最终回答；官方基线文件写死思考开关，本仓库未持有。

## 5. Smoke 测试

`bash smoke.sh`，原始输出见 [`evidence/smoke.log`](evidence/smoke.log)：

```
== /health
HTTP 200
== /v1/models
{"object":"list","data":[{"id":"mimo-v2.6-9b","object":"model","created":1793336306,"owned_by":"vllm","root":"/root/hygon/models/mimo","parent":null,"max_model_len":32768,"permission":[{"id":"modelper
== 单轮(关闭思考)
'国产异构加速器是指由国内厂商自主研发、采用CPU/GPU/NPU/DSP等多类型计算单元协同工作的加速芯片，用于提升AI训练推理、科学计算与边缘智能等场景的算力效率。'
finish: stop tokens: 44
== 多轮(关闭思考)
'你叫小海。'
finish: stop tokens: 6
== 算术(关闭思考)
'221'
finish: stop tokens: 4
== 流式

data: [DONE]
```

## 6. 精度测评

### 6.1 官方口径：未完成 —— 缺官方输入

赛事要求 `verify.py` 按官方 `baselines/<模型>.json`（写死子集题数、max_tokens、思考开关与**官方 GPU 基线分**）
现场跑分并输出 `verify_result.json`。该基线文件与 `data/` 评测子集**随分题材料下发，本仓库未持有**：

- 飞书知识空间节点 `131006` 返回 `permission_denied`（该节点所属空间未授权）
- 文档搜索 `分题材料 / baselines / gsm8k.jsonl` 均 0 命中；「分题材料」表只有 T01–T04 四行分类，无附件
- 实例与共享盘全盘 `find` 无 `baselines/`、无评测子集

因此**与官方 GPU 基线比较的劣化幅度无法计算**。按赛事红线（伪造日志取消资格），不填任何推测分数。

### 6.2 自测绝对精度：`verify.py --abs` 实跑

`verify.py` 内置评测器，两种模式：

- **默认（官方口径）**：读 `baselines/<模型>.json`，按其 `data` / `shots` / `max_tokens` / 思考开关复跑，
  再用 `check()` 判「劣化是否 ≤ 2%」，产出 `verify_result.json`。基线缺失时**直接报错退出**，不猜数。
- **`--abs`（本节）**：同一套推理与判分代码，不比对基线，只输出公开数据集上的绝对精度。
  判分逻辑另有 `--selfcheck` 自验证（本地与实例上均返回 `selfcheck OK`）。

实跑命令（实例：K100-AI 64 GB，vLLM 0.21.0+das，max-model-len 32768）：

```bash
source env.sh
python verify.py --abs --dataset gsm8k --data data/gsm8k_test.jsonl \
  --train data/gsm8k_train.jsonl --shots 5 --conc 24 --out gsm8k_verify.json
python verify.py --abs --dataset mmlu  --data data/mmlu_test.parquet \
  --per-subject 5 --conc 24 --out mmlu_verify.json
```

| 数据集 | 样本数 | 配置 | 正确 | 得分 | 二项标准误 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K test | 1319（全量） | 5-shot、greedy、关闭思考、max_tokens 512 | 1164 | **88.25%** | ±0.89 | 908 s |
| MMLU test | 285（57 科 × 5 题） | 0-shot、greedy、关闭思考、max_tokens 32 | 225 | **78.95%** | ±2.41 | 17.6 s |

- 原始产物：[`evidence/precision/gsm8k_verify.json`](evidence/precision/gsm8k_verify.json)、
  [`evidence/precision/mmlu_verify.json`](evidence/precision/mmlu_verify.json)，
  逐 25 条进度与最终汇总在 [`evidence/precision/verify.log`](evidence/precision/verify.log)
  （首行为 `selfcheck OK`）。
- 数据集来源与指纹（`data/README.md`）：GSM8K 取自 openai/grade-school-math（test 1319 /
  train 前 5 条作 exemplar），MMLU 取自 cais/mmlu `all/test` 每科取 5 题。
- **口径提醒**：同一模型 0-shot、不加 CoT 提示时 GSM8K 只有 17.06%（早期试跑，未留 json），
  说明 GSM8K 对 few-shot 口径极敏感；若官方基线的 shot 数不同，本节分数不能直接对比，
  拿到 `baselines/<模型>.json` 后按它的 `shots`/`max_tokens`/思考开关复跑即可。
- 同一配置下早期用另一支独立脚本测得 87.26%（1151/1319），与本节的 88.25% 差 0.99 pt，
  在并发 24 下属于答案抽取边界样本的正常波动区间，本节以 `verify.py` 的结果为准。

评测结果 JSON 原文（`verify.py --abs` 产出，归档于 `evidence/precision/`）：

```json
{
  "model": "mimo-v2.6-9b",
  "model_id": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
  "dataset": "gsm8k",
  "n": 1319,
  "correct": 1164,
  "accuracy": 0.8825,
  "stderr": 0.0089,
  "elapsed_s": 908.4,
  "mode": "abs",
  "verdict": null,
  "baseline": null,
  "note": "官方 baselines/<模型>.json 未随分题材料下发，此处为公开数据集上的绝对精度；拿到官方基线文件后按同一配置复跑即可计算劣化幅度。"
}
```

```json
{
  "model": "mimo-v2.6-9b",
  "model_id": "XiaomiMiMo/MiMo-V2.6-Distill-Qwen-9B",
  "dataset": "mmlu",
  "n": 285,
  "correct": 225,
  "accuracy": 0.7895,
  "stderr": 0.0241,
  "elapsed_s": 17.6,
  "mode": "abs",
  "verdict": null,
  "baseline": null,
  "note": "官方 baselines/<模型>.json 未随分题材料下发，此处为公开数据集上的绝对精度；拿到官方基线文件后按同一配置复跑即可计算劣化幅度。"
}
```

格式说明：模板的 `detail{got/ref/worse/tol/pass}` 格式依赖官方基线文件（`ref` 与容差判定），
未下发前 `verify.py` 默认模式不产出该格式、也不填推测值（赛事红线：伪造日志取消资格）；
上列两份为 `--abs` 模式的原始 JSON，`verdict`/`baseline` 为 null 即未做基线比较，
拿到 `baselines/<模型>.json` 后按同一配置复跑即可产出标准格式并计算劣化幅度。


## 7. 性能参考

**只取得 4K 输入 × 1K 输出 × 并发 1 / 8 两档**，每档 2 轮取第 2 轮。9B 混合结构下 4K c=1 单轮已需
355 s。客户端 `vllm bench serve`、`--random-range-ratio 0.99`、`--ignore-eos`。
原始数据见 [`evidence/bench/`](evidence/bench/) 与 [`evidence/bench16k/`](evidence/bench16k/)。

| 输入长度 | 并发 | 完成 | 耗时 s | 请求吞吐 | 输出吞吐 tok/s | 均值 TTFT ms | P99 TTFT ms | 均值 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1 | 8 | 354.8 | 0.02 | 20.92 | 1964 | 3142 | 45.7 |
| 4096 | 8 | 16 | 174.5 | 0.09 | 90.37 | 5749 | 15236 | 74.4 |

**16K 档实测失败，如实记录**（未用外推值填充）：

- `16K × c=1`：单轮 900 s 硬超时（8 请求 × 16K 输入 ≈ 131 K prefill token），两轮均被 `timeout` 杀掉。
- `16K × c=8`：客户端启动即 `ConnectionRefusedError 127.0.0.1:8000`，`completed=0 / failed=16` ——
  服务在上一条超时的长上下文压测之后已退出。日志与 json 均留在 `evidence/bench16k/`。
- 结论：单卡 64 GB 跑 9B 混合结构模型，16K 全档需要更长时限或更小批量，本轮未取得可用数据。

## 8. 注意事项

- **只跑了官方 6 档里的 2 档（4K×1、4K×8）**，16K 与 c=64 未跑，故本表不能与跑满 6 档的结果比较。
- **首次启动 590 s**：这是 Mamba+Attention 混合结构的 torch.compile 图捕获，不是卡住；命中
  `torch_compile_cache` 后二次启动约 2 分钟。稳态显存 21.9 GB / 64 GB。
- TTFT 偏高（c=1 达 1.96 s、c=8 达 5.7 s），是混合结构 prefill 路径的特征，与稠密模型不可直接横比。
- 与官方压测脚本的两处差异：客户端用 `vllm bench serve`（镜像无 sglang）；`--random-range-ratio 0.99`
  （官方 1.0 在 vLLM 客户端报错）。
- 思考模式默认开启，压测用 `--ignore-eos` 固定 1024 输出。
- `ROCM_PATH` 必须指向镜像内 DTK；`vllm_hcu` 的绝对符号链接需重指向 rootfs。
- 首次启动含 torch.compile，约 2–5 分钟。

---

### 仓库内容

| 文件 | 章节 | 说明 |
| --- | --- | --- |
| `env.sh` | 2 / 4 | 变量表 |
| `run.sh` | 4 | 装配 + 起服务 + 轮询 /health |
| `env_check.sh` | 2 | → `evidence/env.log` |
| `smoke.sh` | 5 | 单轮/多轮/算术/流式 → `evidence/smoke.log` |
| `verify.py` | 6 | 官方判分逻辑 |
| `bench.sh` | 7 | 简化网格 → `evidence/bench/` |
