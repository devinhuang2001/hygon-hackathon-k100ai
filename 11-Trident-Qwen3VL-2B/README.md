# OpenBMB/Trident-Qwen3VL-2B on 异构加速卡 BW + SGLang 0.5.18

> 赛题：T02 视觉多模态理解模型（VL / VLA）SGLang / vLLM 适配 · 嵌入模型
> 计件粒度：`Trident-Qwen3VL-2B × 异构加速卡 BW × SGLang`
> 实测日期：2026-11-22　平台：SCNet 超算互联网 Notebook（异构加速卡 BW × 1）

## 1. 简介

Trident-Qwen3VL-2B 是 OpenBMB 的**混合模态检索嵌入模型**：把文本、图像、以及“图像+文本”融合内容映射到同一向量空间，一条文本 query 可以对纯文本、纯图片、图文三类文档统一排序。权重由 `Qwen/Qwen3-VL-2B-Instruct` 微调（LoRA 已合并），2.13B 参数、BF16、4.0 GB，单卡即可部署。

- 权重：`openbmb/Trident-Qwen3VL-2B`（16 个文件 / 4.0 GB，含官方 `trident_embedder.py`）
- 输出：**2048 维、L2 归一化**向量；query 与 doc 使用**不同 instruction**；`mmemb_model.json` 指定 `pooling=last_token`、`normalize=true`、`add_generation_prompt=true`
- 验收口径（T02）：跑通「图文输入 → 向量输出（embedding 接口）」，用公示检索子集的检索指标；**本件免性能要求**

**参考文档**

- 模型卡与权重：https://huggingface.co/openbmb/Trident-Qwen3VL-2B
- 官方代码：https://github.com/OpenBMB/Trident
- sglang 文档：https://docs.sglang.io/
- 海光镜像清单：https://github.com/HYGON-AI/inference-cookbook-das/blob/main/docs/model-deployment/docker_images.md

**提交仓库（光合开发者社区，评审以此仓为准）**：https://developer.sourcefind.cn/codes/g_x8d7w4tr/Trident-Qwen3VL-2B-BW
（镜像：https://github.com/devinhuang2001/hygon-hackathon-k100ai/tree/main/11-Trident-Qwen3VL-2B）

**一键复现**

```bash
git clone https://developer.sourcefind.cn/codes/g_x8d7w4tr/Trident-Qwen3VL-2B-BW.git
cd Trident-Qwen3VL-2B-BW
bash run.sh                      # 起服务（内含 DTK/头文件/多模态全部参数）
python3 smoke.py                 # 三种输入形态
python3 diag_mm_input.py         # 图像是否真进模型（三组余弦）
python3 ref_text_match.py        # 文本与官方实现的一致性
python3 eval_ndcg.py --backend ref      # 官方 embedder 基线
python3 eval_ndcg.py --backend sglang   # 本件
```

## 2. 验证环境

| 项 | 值 |
|---|---|
| 主机 | 128 核 / 503 GB 内存 / overlay 1.8 T |
| 加速卡 | **BW × 1，64.0 GiB，`gfx936`**（`hy-smi --showproductname` → `Card Series: BW`），PwrCap 1000 W |
| DTK | 26.04（镜像内用户态），HIP 6.3.26113 |
| 开发镜像 | `harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10`，manifest `sha256:73656c45c1b6301ad459056a6e5d520d26039abe5364116f4a9c3a59da3ceae6` |
| 运行方式 | Notebook 无 Docker：`skopeo` 拉 OCI + `umoci unpack`，用**宿主 `/usr/bin/python3.10`** 加载镜像内 dist-packages 与 DTK 用户态 |
| 软件版本 | torch 2.11.0（DTK 版）、sglang 0.5.18、transformers 5.17.0 |
| 精度 | BF16，未量化；权重未改动 |
| 服务映射 | `127.0.0.1:19100`（Notebook 内回环，未外发端口） |

服务端启动关键日志（原始输出见 `evidence/server3.log`）：

```text
server_args=... is_embedding=True, enable_multimodal=True,
    json_model_override_args='{"architectures": ["Qwen3VLForConditionalGeneration"]}', dtype='bfloat16'
Multimodal data loading enabled with 16 worker threads (auto).
Load weight end. elapsed=11.76 s, type=Qwen3VLForConditionalGeneration,
    avail mem=58.38 GB, mem usage=4.18 GB.
max_total_num_tokens=136885, max_running_requests=2048, context_len=262144,
    available_gpu_mem=41.67 GB
The server is fired up and ready to roll!
```

## 3. 适配优化方案

### 3.1 基础镜像

用赛事指定的 sglang 0.5.18 镜像。Notebook 无 Docker，故 `skopeo copy` 到 OCI layout、`umoci unpack` 解包，再用宿主 python 加载镜像内的 dist-packages 与 DTK 用户态：`ROCM_PATH` 指向镜像 `opt/dtk`，`LD_LIBRARY_PATH` 额外加镜像的 `.hyhal/rocm_smi/lib` 与宿主 `/opt/hyhal/lib`（否则 `librocm_smi64.so.2` 找不到）。

### 3.2 软件包升级

**无。** 未安装任何额外 pip 包，全部使用镜像自带版本（torch 2.11.0 / sglang 0.5.18 / transformers 5.17.0）。

### 3.3 代码改动

不改权重、不改 tokenizer、不改采样语义。以下四处都在**启动参数与运行环境**层面：

**(1) 架构名覆盖。** 权重 `config.json` 声明 `architectures: ["Qwen3VLForEmbedding"]`，该类只存在于仓库自带的 `trident_embedder.py`，transformers 与 sglang 均未注册，`--is-embedding` 会在 `get_mm_processor_cls` 阶段直接抛 `ValueError`。用启动参数把它指到 sglang 已实现的 `qwen3_vl.py`：

```bash
--json-model-override-args '{"architectures": ["Qwen3VLForConditionalGeneration"]}'
```

sglang 的 `qwen3_vl.py` 内部即 `Pooler(pooling_type=PoolingType.LAST, normalize=True)`，与权重 `mmemb_model.json` 的 `last_token + normalize` 一致 —— 这是该路线成立的前提。

**(2) 宿主缺 Python 头文件。** Triton 现场编译 `hip_utils.c` 需要 `Python.h`，缺失时 Scheduler 抛 `CalledProcessError` 并崩服。把镜像头文件挂进 include 路径即可（不装包）：

```bash
C_INCLUDE_PATH=$ROOTS/usr/include/python3.10:$HEADERS
CPLUS_INCLUDE_PATH=$C_INCLUDE_PATH     # $HEADERS/x86_64-linux-gnu/python3.10 -> 镜像同名目录
```

**(3) 图像输入必须用 sglang 自己的形态。** `/v1/embeddings` 若用 OpenAI messages（`role`/`content`/`image_url`），**不报错但静默丢图**：带图与不带图余弦 0.999945、两张不同图 0.999976。必须写成 `MultimodalEmbeddingInput`：

```json
{"model": "trident-qwen3vl-2b", "input": [{"text": "<doc instruction>", "image": "data:image/png;base64,..."}]}
```

改对之后：带图 vs 不带图 0.644、两张不同图 0.692、同输入重复 0.99996（`diag_mm_input.py`）。另外必须显式加 `--enable-multimodal`（默认未开）。

**(4) 文本输入要走官方 chat 模板。** 官方 prompt 配置是 `use_system_role=true` + `add_generation_prompt=true`，池化取 assistant 前缀之后的位置；sglang 的 messages 输入不补该前缀，同一句话只能到 0.782 余弦。**先用官方 tokenizer 渲染成字符串再送 `input:"<string>"`，余弦 0.999901**（`ref_text_match.py`）。

### 3.4 升级步骤

```bash
mkdir -p /root/hygon/{trident/evidence,models,bundle,oci}
skopeo copy --retry-times 3 \
  docker://harbor.sourcefind.cn:5443/hcu/admin/base/sglang:0.5.18-ubuntu22.04-dtk26041-py3.10 \
  oci:/root/hygon/oci/sglang:0.5.18
apt-get install -y umoci
umoci unpack --rootless --image /root/hygon/oci/sglang:0.5.18 /root/hygon/bundle/sglang
HF_ENDPOINT=https://hf-mirror.com hf download openbmb/Trident-Qwen3VL-2B \
  --local-dir /root/hygon/models/Trident-Qwen3VL-2B
bash /root/hygon/trident/run.sh
```

## 4. 服务启动脚本

`run.sh` 全文（光合仓根目录 `run.sh`；GitHub 镜像在 `11-Trident-Qwen3VL-2B/kit/run.sh`；实例上为 `/root/hygon/trident/run.sh`）：

```bash
R=/root/hygon/bundle/sglang/rootfs
export ROCM_PATH=$R/opt/dtk; export DTKROOT=$ROCM_PATH; export HIP_PATH=$ROCM_PATH/hip
export PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$ROCM_PATH/hip/bin:/opt/hyhal/bin:$PATH
export LD_LIBRARY_PATH=$ROCM_PATH/lib:$ROCM_PATH/lib64:$ROCM_PATH/hip/lib:$R/opt/dtk/.hyhal/rocm_smi/lib:/opt/hyhal/lib:/opt/hyhal/lib64
export PYTHONPATH=$R/usr/local/lib/python3.10/dist-packages:$R/usr/lib/python3/dist-packages
PH=/root/hygon/trident/python-headers
mkdir -p "$PH/x86_64-linux-gnu"
ln -sfnT "$R/usr/include/x86_64-linux-gnu/python3.10" "$PH/x86_64-linux-gnu/python3.10"
export C_INCLUDE_PATH="$R/usr/include/python3.10:$PH"
export CPLUS_INCLUDE_PATH="$C_INCLUDE_PATH"
exec /usr/bin/python3.10 -m sglang.launch_server \
  --model-path /root/hygon/models/Trident-Qwen3VL-2B \
  --served-model-name trident-qwen3vl-2b \
  --is-embedding --enable-multimodal \
  --json-model-override-args '{"architectures": ["Qwen3VLForConditionalGeneration"]}' \
  --host 127.0.0.1 --port 19100 --trust-remote-code --dtype bfloat16 \
  --attention-backend triton --disable-cuda-graph --mem-fraction-static 0.30
```

## 5. Smoke 测试

`smoke.py` 三种输入形态（2048 维、L2≈1）：

| 测试 | 输入 | 返回 |
|---|---|---|
| 纯文本 query | 文本 | 2048 维，L2 0.9984 |
| 单图 doc | ChartQA 图 PNG（data URI） | 2048 维，L2 0.9982 |
| 图文融合 doc | 图 + 文本 | 2048 维，L2 0.9986 |

图像是否真的进入模型（`diag_mm_input.py`，同一 instruction）：

| 比较 | 余弦 |
|---|---:|
| 带图 vs 不带图 | 0.6440 |
| 两张不同图 | 0.6918 |
| 同一输入重复两次 | 0.99996 |

文本路径与官方 `TridentEmbedder` 的逐条一致性：

| 喂法 | 余弦 vs 官方 |
|---|---:|
| chat 模板渲染 + assistant 前缀 | **0.999901** |
| 无 assistant 前缀 | 0.782344 |
| 丢 system instruction | 0.925447 |
| 裸文本 | −0.018035 |

## 6. 精度测评

### 6.1 数据集与口径

用模型作者公开的 **`OpenBMB/Trident-Eval`** 的 **ChartQA 子集（Origin 设定）**：1509 篇图像文档 × 1250 条文本 query，每 query 一个相关文档。指标 NDCG@10（IDCG=1，相关文档排名 r 得 `1/log2(r+2)`）与 Recall@{1,5,10}。两种后端跑**同一份数据、同一套 instruction、同一套模板**：

| 后端 | NDCG@10 | Recall@1 | Recall@5 | Recall@10 |
|---|---:|---:|---:|---:|
| 官方 `TridentEmbedder`（transformers，**GPU 参考基线**） | 69.49 | 52.48 | 78.00 | 87.52 |
| **本件 SGLang `/v1/embeddings`（BW）** | **70.30** | **54.08** | **79.52** | **87.68** |
| 相对基线 | **+1.17%** | +1.60 | +1.52 | +0.16 |

结论：**较官方实现没有劣化**（验收要求劣化 ≤2%，此处 +1.17%，属正常波动；差异来自 doc 侧 chat 模板渲染细节，不是权重或数值精度问题）。

原始证据：`evidence/eval_chartqa_ref.json`、`evidence/eval_chartqa_sglang.json`；逐条向量由 `eval_ndcg.py` 在 `evidence/emb_chartqa_<backend>.npz` 落盘（1509×2048 文档 + 1250×2048 query，运行时重建）。

### 6.2 官方材料说明与未完成项

- **赛事自带的检索子集未下发**：官方「分题材料」表只写 T02 会提供「检索子集 + 评测脚本 + 官方基线文件」，没有附下载入口，赛事文档正文也搜不到。因此本报告用模型作者公开、可匿名复现的 Trident-Eval 作为「公示」数据，并把**官方实现本身当基线**做同口径对比；官方子集到手后换数据目录重跑 `eval_ndcg.py` 即可。
- **只测了 1 个视觉文档子集**：其余（DocVQA 1.1 GB、InfoVQA 1.1 GB、Dude 9.4 GB、Google_WIT 2.4 GB 等）未跑。模型卡上「视觉文档 6 集合 Origin 平均 78.28」是六集合平均，不能与单集合 69.49 直接比较，故不引用为基线。
- **Mix 设定未测**：`mix_corpus.jsonl`（图文融合文档）已在数据中，本次只跑 Origin（纯图文档）。

## 7. 性能参考

单条请求（batch=1，含 HTTP 往返；同机另有其他负载，本服务显存上限压到 0.30）：

| 场景 | 次数 | 中位延迟 | 最小 / 最大 |
|---|---:|---:|---:|
| 纯文本（每次新 prompt） | 5 | **38 ms** | 38 / 43 ms |
| 纯文本（同输入重复） | 5 | 35 ms | 35 / 38 ms |
| 单图 doc（每次新 prompt） | 5 | **168 ms** | 119 / 193 ms |
| 单图 doc（同输入重复） | 5 | 149 ms | 118 / 200 ms |

本件**免性能要求**，以上仅作环境标定：未做并发压测，也未与官方 GPU 基线做性能对比。权重加载 11.76 s、常驻 4.18 GB，起服到 ready 约 65 s。

## 8. 注意事项

1. **图像输入形态是本件最大的坑**：用 OpenAI messages 写法不会报错，但图像被静默丢弃（余弦 0.99995），检索指标会直接崩到 NDCG 0.49。必须用 `{"text","image"}` 且开 `--enable-multimodal`。
2. **必须覆盖 architectures**：不改 config 直接 `--is-embedding` 会因 `Qwen3VLForEmbedding` 未注册而拒绝启动。
3. **宿主需要 Python 头文件**：否则 Triton 编译 `hip_utils.c` 失败、Scheduler 崩服，且日志里要翻到 `CalledProcessError` 才能定位。
4. **同卡共存**：实例上另有其他负载长期运行，本服务把 `--mem-fraction-static` 压到 0.30，未干预其他进程；满负载吞吐未测。
5. **可复现边界**：69.49 / 70.30 均在本机 BW 上实测，脚本、日志、向量齐全；换机后若 chat 模板版本不同，需按 `ref_text_match.py` 重新对齐 assistant 前缀。
