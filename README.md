# 国芯国模黑客松 · 提交仓库（K100-AI / vLLM 0.21.0）

参赛者：黄相潼　超算互联网账号：18998110356

本仓库收录在 **SCNet 超算互联网 · 异构加速卡 AI-64GB（K500SM_AI 64GB，按 K100-AI 计件）**
上完成适配与实测的模型包。每个子目录是一个**可独立验收的件**（模型 × 卡型 × 框架），
目录内的 `README.md` 即该件按官方《技术报告模板》8 章节填写的完整技术报告，
目录内的脚本与 `env.log` / `server.log` / `smoke.log` / `bench/` 是报告中每一个数字的原始出处。

## 一、本次提交的件

| # | 模型 | 赛题 | 框架 | 卡型 | 部署 | Smoke | 性能实测 | 精度 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [01](01-Qwen3-ASR-1.7B/README.md) | **Qwen3-ASR-1.7B** | **T04 语音识别（计件）** | vLLM 0.21.0+das | K100-AI | ✅ | ✅ 中英转写逐字正确 | ✅ 3 档并发：21.97 → 208.23 audio-s/s | ⏳ 待官方基线 |
| [02](02-Qwen3-ASR-0.6B/README.md) | **Qwen3-ASR-0.6B** | **T04 语音识别（计件）** | vLLM 0.21.0+das | K100-AI | ✅ | ✅ 中英转写逐字正确 | ✅ 3 档并发：37.04 → 225.05 audio-s/s | ⏳ 待官方基线 |
| [07](07-MiMo-V2.6-Distill-Qwen-9B/README.md) | **MiMo-V2.6-Distill-Qwen-9B** | **T01 文本 LLM（计件）** | vLLM 0.21.0+das | K100-AI | ✅ | ✅ 单轮/多轮/算术/流式 | ✅ 4K×1、4K×8：20.92 / 90.37 tok/s | ✅ 自测 GSM8K 87.26% / MMLU 78.95% |

另有 4 个非赛题模型（`03`–`06`，见下表），作为同卡型、同框架下的横向性能参照一并存放，
可用于后续按同一负载复测对比，不计件。

| # | 模型 | 规模 | 4K 输出吞吐 c=1 → c=8（tok/s） | 备注 |
| --- | --- | --- | --- | --- |
| [03](03-Qwen3-0.6B/README.md) | Qwen3-0.6B | 0.6B | 97.59 → 329.99 | 官方 6 档全覆盖，16K×64 单轮 >1200 s 超时 |
| [04](04-Qwen3-1.7B/README.md) | Qwen3-1.7B | 1.7B | 64.56 → 269.29 | 4K / 16K × c1 / c8 |
| [05](05-Qwen3-4B/README.md) | Qwen3-4B | 4B | 40.26 → 170.28 | 4K / 16K × c1 / c8 |
| [06](06-Qwen3-8B/README.md) | Qwen3-8B | 8B | 21.28 → 97.29 | 仅 4K 档，单轮 349 s / 162 s |

## 二、统一验证环境

| 项 | 值 |
| --- | --- |
| 平台 | 超算互联网 SCNet Notebook（`ksai.scnet.cn`） |
| 节点 | Hygon C86 7185 32C / 503 GB RAM |
| 加速卡 | 1× K500SM_AI 64 GB（120 SM）→ 官方口径 **K100-AI** |
| 镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/vllm:0.21.0-ubuntu22.04-dtk2604-py3.10` |
| 框架 | vLLM `0.21.0+das.dtk2604`（含 `vllm_hcu` 国产加速插件） |
| DTK / torch | DTK `6.3.26113` / torch `2.10.0+das.opt1.dtk2604`（HIP 6.3.26113） |
| Python | 3.10.21 |

整套环境**没有容器运行时**：SCNet 该实例不提供 docker/podman，镜像 24 层（7.87 GB）从
`harbor.sourcefind.cn` 匿名拉取后在 Notebook 内解包为 `/root/hygon/rootfs`，
用宿主内核视图 + 镜像内 Python 包与 DTK 运行。三处必须的运行时装配（符号链接重指向、
补齐 `/opt/hyhal`、`ROCM_PATH` 指向镜像内 DTK）见各包 README 第 3 节。

## 三、目录结构

```
<本仓库>/
├── 01-Qwen3-ASR-1.7B/     # 一个可独立验收的件
│   ├── README.md          # 技术报告，官方模板 8 章节
│   ├── env.sh             # 模型 / 镜像 / 框架 / 卡型 / 并行 / 启动参数
│   ├── env_check.sh       # 采集卡型与组件版本 → env.log（第 2 节）
│   ├── run.sh             # 装配运行时 + 重放装配 + 起服务 → server.log（第 4 节）
│   ├── smoke.sh           # 健康检查 + 单轮 / 多轮 / 流式 → smoke.log（第 5 节）
│   ├── verify.py          # 精度验证 → verify_result.json（第 6 节）
│   ├── bench.sh           # 性能压测 6 档 × 2 轮 → bench/（第 7 节）
│   ├── asr_bench.py       # T04 专用：ASR 并发吞吐（实时倍率 / RTF）
│   ├── data/              # 公开测试音频
│   ├── bench/             # 两轮压测的 jsonl、各档日志、summary.md
│   └── env.log / server.log / smoke.log / evidence/
└── 03…07/                 # 同上
```

## 四、复现步骤

```bash
# 0) 在本实例内准备 rootfs（一次性）：harbor 匿名拉 24 层 → 解包
#    详见各包 README 第 3 节 3.4
cd 01-Qwen3-ASR-1.7B
bash run.sh          # 起服务，轮询 /health 就绪
bash env_check.sh    # 采集版本 → env.log
bash smoke.sh        # 业务 smoke → smoke.log
source env.sh && python verify.py    # 精度（需官方 baselines/ 与 data/）
bash bench.sh        # 性能 6 档 × 2 轮 → bench/
```

## 五、与官方模板的差异（逐条说明，未做静默改动）

1. **压测客户端**：官方 `bench.sh` 模板用 `sglang.benchmark.serving`；本镜像不含 sglang，
   改用镜像自带的 `vllm bench serve`，负载参数（输入 4K/16K、输出 1K、并发 1/8/64、每档两轮取第二轮）保持一致。
2. **`--random-range-ratio`**：官方写 1，vLLM 客户端在该取值下报错（min input tokens 为 0），
   实测用 0.99，输入长度分布差异 <1%。
3. **`--ignore-eos`**：保证输出长度固定 1024，与「1K 输出」口径一致。
4. **16K×64 档**：单卡单轮实测 >1200 s（0.6B 即超时，128 请求 × 16K 输入 ≈ 2 M prefill token），
   批量跑时按模型规模裁剪，缺档在各包第 7 节逐条注明，未用估算值填充。
5. **端口**：官方模板 `PORT=30000`，本环境统一用 `8000`（脚本内由 `env.sh` 的 `PORT` 控制，改一行即可对齐）。

## 六、已知缺口：精度章节待官方基线

赛事的精度门槛要求 `verify.py` 按官方 `baselines/<模型>.json`（写死子集题数、max_tokens、
思考开关与**官方 GPU 基线分**）现场跑分并输出 `verify_result.json`。这套文件随分题材料下发，
本次未能取得：

- 飞书知识空间节点 `131006` 返回 `permission_denied`（该节点所属空间未授权）
- 文档搜索 `分题材料 / baselines / gsm8k.jsonl` 均 0 命中
- 实例与共享盘全盘 `find` 无 `baselines/`、无评测子集

因此各包第 6 节**留空并写明原因**，未填任何推测分数——赛事红线明确「伪造日志取消资格」。
`verify.py` 已实现官方判分逻辑并支持 `--selfcheck`；把官方 `baselines/<模型>.json` 与
`data/` 放入对应目录后执行 `source env.sh && python verify.py` 即可产出 `verify_result.json`。

例外是 **07（T01 MiMo-9B）**：该赛道门槛点名的 GSM8K / MMLU 有公开数据集，因此用写死配置
实跑了绝对分 —— GSM8K test 全量 5-shot **87.26%**（1151/1319，±0.92）、MMLU test 每科 5 题
**78.95%**（225/285，±2.41），脚本 `07-MiMo-V2.6-Distill-Qwen-9B/precision.py`，原始结果在
`evidence/precision/`。**这只是绝对分**：官方 GPU 基线文件仍未下发，劣化幅度依旧算不出来。

## 七、声明

- 未修改模型权重、结构、tokenizer 与采样语义；未硬编码任何答案。
- 所有性能数字来自本仓库 `bench/round2_*.json` 原始输出，未做手工修饰。
- 精度分数缺失是取材限制，不是选择性省略。
