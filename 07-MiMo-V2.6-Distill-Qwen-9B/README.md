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

### 6.2 自测绝对精度（公开数据集，配置写死）

为不让本节空白，用公开数据集与写死配置在本卡上实跑，给出**绝对分**供验收方比对。

**这两行数字不是 `verify.py` 跑出来的**——`verify.py` 缺 `baselines/` 会直接退出，只保留了判分逻辑
与 `--selfcheck`。实际执行的是自写脚本 [`precision.py`](precision.py)，只算绝对分、不比对基线；
解码口径沿用官方要求的 greedy / 关闭思考 / 固定 max_tokens：

| 数据集 | 样本数 | 配置 | 正确 | 得分 | 二项标准误 | 耗时 |
| --- | --- | --- | --- | --- | --- | --- |
| GSM8K test | 1319（全量） | 5-shot、greedy、max_tokens 512 | 1151 | **87.26%** | ±0.92 | 833 s |
| MMLU test | 285（57 科 × 5 题） | 0-shot、greedy、max_tokens 32 | 225 | **78.95%** | ±2.41 | 18 s |

- 原始结果：[`evidence/precision/gsm8k_result.json`](evidence/precision/gsm8k_result.json)、
  [`evidence/precision/mmlu_result.json`](evidence/precision/mmlu_result.json)；
  各抽 8 条带原样的核对样本在 `*.samples.txt`（GSM8K 输出为带 `####` 的完整 CoT，判分未被格式污染）。
- 数据集来源与指纹（`data/README.md`）：GSM8K 取自 openai/grade-school-math（test 1319 /
  train 前 5 条作 exemplar），MMLU 取自 cais/mmlu `all/test` 每科取 5 题。
- **口径提醒**：同一模型 0-shot、不加 CoT 提示时 GSM8K 只有 17.06%（早期试跑，未留 json），
  说明 GSM8K 对 few-shot 口径极敏感；若官方基线的 shot 数不同，本节分数不能直接对比，
  拿到 `baselines/<模型>.json` 后按它的 `shots`/`max_tokens`/思考开关复跑即可。


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
