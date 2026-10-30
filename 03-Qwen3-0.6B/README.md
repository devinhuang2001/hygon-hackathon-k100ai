# Qwen3-0.6B on K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0

> 按《国芯国模黑客松 · 技术报告模板》8 章节填写，作为提交仓库的 `README.md`。
> **赛题归属说明**：`Qwen3-0.6B` **不在**本赛季 18 个赛题模型清单内（T01 为 Kimi-Linear-48B-A3B / MiMo-V2.6-Distill-Qwen-9B）。
> 本包用途是「同栈文本模型的可复现验证样例」——验证本文档描述的环境、脚本与官方压测口径在 K100-AI 上成立，
> 可直接套用到 T01 模型上，**不构成计件**。

## 1. 简介

记录 `Qwen/Qwen3-0.6B` 在 `K100-AI（K500SM_AI 64GB）+ vLLM 0.21.0`（bf16）的部署与实测。

Qwen3-0.6B 是 Qwen3 系列最小规格的稠密语言模型（28 层），支持思考/非思考两种模式，单卡部署。

与官方部署方式的核心差异：未升级包、未改模型代码；两处运行时装配（符号链接、`ROCM_PATH`）见第 3 节。
权重来自本机共享模型库（`/root/public_data/model/admin/Qwen/Qwen3-0.6B`），非重新下载。

- 权重：https://www.modelscope.cn/models/Qwen/Qwen3-0.6B
- 镜像：`harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10`

## 2. 验证环境

| 项 | 值 |
| --- | --- |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` |
| 卡型 | K500SM_AI 64 GB × 1（计 K100-AI），120 SM |
| 节点 | SCNet Notebook，Hygon C86 7185 32C / 503 GB RAM |
| Python / torch | 3.10.21 / 2.10.0+das.opt1.dtk2604.2608141812（HIP 6.3.26113） |
| triton / vllm | 3.6.0+das185.dtk2604 / 0.21.0+das.dtk2604 |
| transformers | 5.5.0 |
| 模型路径 | `/root/public_data/model/admin/Qwen/Qwen3-0.6B` |
| 端口 | 8000 |

完整输出见 [`evidence/env.log`](evidence/env.log)。

## 3. 适配优化方案

3.1 基础镜像同上。3.2 软件包升级：无。

3.3 代码改动：无模型代码改动，仅两处运行时装配——
(1) rootfs 内 `vllm/.../fused_moe/modular_kernel.py` 绝对符号链接失效，改指向 `$ROOTFS`；
(2) `ROCM_PATH` 指向镜像内 DTK（含 triton 所需的 `aillvm`），否则 kernel 编译报 `HSACOError`。

3.4 升级步骤：解包镜像 → `bash run.sh` → `bash env_check.sh` → `bash smoke.sh` → `bash bench.sh`。

## 4. 服务启动脚本

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /root/public_data/model/admin/Qwen/Qwen3-0.6B \
  --served-model-name qwen3-0.6b \
  --trust-remote-code --dtype bfloat16 \
  --max-model-len 32768 --gpu-memory-utilization 0.90 \
  --host 127.0.0.1 --port 8000
```

参数理由同 T04 包。**思考模式默认开启**：Qwen3 的 chat template 默认 `enable_thinking=true`，128 token 的 smoke
会被 `<think>` 段吃满（首轮实测 `finish_reason=length`、未给出最终回答），所以第 5 节统一用
`chat_template_kwargs={"enable_thinking": false}` 取最终回答。官方基线文件写死思考开关，本仓库未持有，
跑精度时必须与基线文件保持一致。

## 5. Smoke 测试

`bash smoke.sh`，原始输出见 [`evidence/smoke.log`](evidence/smoke.log)：

```
== /health            HTTP 200
== 单轮(关闭思考)
'国产异构加速器是一种结合了CPU和GPU的新型计算架构，广泛应用于人工智能、高性能计算等领域。'
finish: stop tokens: 25
== 多轮(关闭思考)
'你好，我是小海。'
finish: stop tokens: 7
== 算术(关闭思考)
'13 × 17 = 221'
finish: stop tokens: 12
== 流式
data: [DONE]
```

```bash
curl http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" \
  -d '{"model":"qwen3-0.6b","messages":[{"role":"user","content":"13*17 等于多少？只回答数字。"}],
       "temperature":0,"max_tokens":64,"chat_template_kwargs":{"enable_thinking":false}}'
```

单轮、算术（13×17=221）均正确；多轮能复述用户名字，但自称「小海」，属 0.6B 规格的角色混淆，不影响接口链路验证。

## 6. 精度测评

**未完成 —— 缺官方输入。** 需要 `baselines/Qwen__Qwen3-0.6B.json`（写死 GSM8K/MMLU 子集题数、max_tokens、
思考开关与 GPU 基线分）+ `data/` 子集，随分题材料下发，本仓库未持有。`verify.py` 已实现官方判分逻辑
（容差 2%、只罚劣化、`--selfcheck` 通过）。按红线不臆造分数。

## 7. 性能参考

官方 6 档口径（4K/16K 输入 × 1K 输出 × 并发 1/8/64，2 轮取第 2 轮）。
客户端为 `vllm bench serve`（镜像无 sglang），`--random-range-ratio 0.99`、`--ignore-eos`，差异见第 8 节。
原始数据见 [`evidence/bench/`](evidence/bench/)（每档 json + log）。

| 输入长度 | 并发 | 完成请求 | 耗时 s | 输出吞吐 tok/s | 请求吞吐 req/s | 均值 TTFT ms | P99 TTFT ms | 均值 TPOT ms |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 4096 | 1 | 8 | 76.1 | 97.59 | 0.11 | 76.0 | 110.1 | 10.18 |
| 4096 | 8 | 16 | 47.8 | 329.99 | 0.33 | 196.2 | 376.2 | 19.23 |
| 4096 | 64 | 128 | 271.7 | 482.51 | 0.47 | 3499.4 | 15550.4 | 115.21 |
| 16384 | 1 | 8 | 113.6 | 65.36 | 0.07 | 218.9 | 329.8 | 15.26 |
| 16384 | 8 | 16 | 104.8 | 150.42 | 0.15 | 521.8 | 1134.8 | 45.65 |
| 16384 | 64 | — | — | 未完成（超时） | — | — | — | — |

- 4K 档随并发从 97.6 → 482.5 tok/s（4.9×），但 P99 TTFT 从 110 ms 涨到 15.5 s，说明 64 并发下排队严重。
- 16K 档吞吐明显低于 4K（65 vs 98 tok/s @c=1），prefill 成为主要成本。
- **16K×64 未完成**：单轮超过脚本 1200 s 上限被终止（128 请求 × 16K 输入，prefill 总量约 2M token）。
  真实耗时未测到，不做估计。

## 8. 注意事项

- 三处与官方压测脚本的差异，逐条记录（否则数据不可比）：客户端用 `vllm bench serve` 而非
  `sglang.benchmark.serving`；`--random-range-ratio 0.99`（官方 1.0 在 vLLM 客户端报错）；加 `--ignore-eos`。
- **16K×64 档在单卡 K100-AI 上未跑完**，若要交这一档需放宽超时或降低并发，不能沿用本表的超时设置。
- 思考模式默认开启，压测用 `--ignore-eos` 固定 1024 输出，因此吞吐受思考段影响；对比不同模型时须统一开关。
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
| `bench.sh` | 7 | 官方 6 档 → `evidence/bench/` |
