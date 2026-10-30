# Qwen3-4B on K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0

> 按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> **赛题归属**：本模型不在本赛季 18 个赛题模型清单内，本包是同栈文本模型的可复现验证样例，
> 用于证明本文档所述环境、脚本与压测口径在 K100-AI 上成立，**不构成计件**。

## 1. 简介

记录 `Qwen/Qwen3-4B` 在 `K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0`（bf16）的部署与实测。

Qwen3-4B 是 Qwen3 系列的中等规格稠密语言模型（36 层），支持思考/非思考两种模式，单卡部署。

与官方部署方式的核心差异：未升级任何包、未改模型代码；仅两处运行时装配（符号链接改指向、`ROCM_PATH`
指向镜像内 DTK），见第 3 节。权重取本机共享模型库 `/root/public_data/model/admin/Qwen/Qwen3-4B`。

- 权重：https://www.modelscope.cn/models/Qwen/Qwen3-4B
- 镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` |
| 卡型 | K500SM_AI 64 GB × 1（计 K100-AI），120 SM |
| 节点 | SCNet Notebook，Hygon C86 7185 32C / 503 GB RAM |
| Python / torch | 3.10.21 / 2.10.0+das.opt1.dtk2604.2608141812（HIP 6.3.26113） |
| triton / vllm | 3.6.0+das185.dtk2604 / 0.21.0+das.dtk2604 |
| 模型路径 | `/root/public_data/model/admin/Qwen/Qwen3-4B` |

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
  --model /root/public_data/model/admin/Qwen/Qwen3-4B \
  --served-model-name qwen3-4b \
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
{"object":"list","data":[{"id":"qwen3-4b","object":"model","created":1793327964,"owned_by":"vllm","root":"/root/public_data/model/admin/Qwen/Qwen3-4B","parent":null,"max_model_len":32768,"permission":
== 单轮(关闭思考)
'国产异构加速器是指由中国自主研发的、用于提升计算性能的非统一架构处理器，能够高效处理并行计算任务，广泛应用于人工智能、大数据、高性能计算等领域。'
finish: stop tokens: 39
== 多轮(关闭思考)
'我叫小海！很高兴认识你！有什么我可以帮你的吗？😊'
finish: stop tokens: 17
== 算术(关闭思考)
'221'
finish: stop tokens: 4
== 流式

data: [DONE]
```

## 6. 精度测评

**未完成 —— 缺官方输入。** 需要 `baselines/<模型>.json`（写死子集题数、max_tokens、思考开关与 GPU 基线分）
与 `data/` 子集，随分题材料下发，本仓库未持有。`verify.py` 已实现官方判分逻辑（容差 2%、只罚劣化、
`--selfcheck` 通过）。按红线不臆造分数。

## 7. 性能参考

简化网格：4K / 16K 输入 × 1K 输出 × 并发 1 / 8，每档 2 轮取第 2 轮。客户端 `vllm bench serve`，
`--random-range-ratio 0.99`、`--ignore-eos`。原始数据见 [`evidence/bench/`](evidence/bench/)。

| 输入长度 | 并发 | 完成 | 耗时 s | 请求吞吐 | 输出吞吐 tok/s | 均值 TTFT ms | P99 TTFT ms | 均值 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1 | 8 | 184.3 | 0.04 | 40.26 | 96 | 129 | 24.8 |
| 4096 | 8 | 16 | 92.6 | 0.17 | 170.28 | 274 | 443 | 36.6 |
| 16384 | 1 | 8 | 231.1 | 0.03 | 32.12 | 255 | 382 | 31.1 |
| 16384 | 8 | 16 | 171.1 | 0.09 | 92.16 | 705 | 1338 | 73.8 |

## 8. 注意事项

- **未跑官方 6 档中的 16K×64**：单轮在单卡上 >20 min（128 请求 × 16K ≈ 2M prefill token），批量跑时省略，
  因此本表不能与跑满 6 档的结果直接比较。
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
