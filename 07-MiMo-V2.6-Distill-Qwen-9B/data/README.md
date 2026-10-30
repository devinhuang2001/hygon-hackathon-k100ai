# 评测数据（未随仓库分发，给出出处与指纹）

官方 `data/` 子集未下发；本节用的是**公开数据集**，按下面的 URL 下载即可复现，不需要向赛事方索取。
文件较大（合计 ~8.4 MB），不放进仓库，下载后放到本目录，文件名保持一致即可直接跑 `precision.py`。

| 文件 | 来源 | 大小 | SHA-256 |
| --- | --- | --- | --- |
| `gsm8k_test.jsonl` | openai/grade-school-math `grade_school_math/data/test.jsonl`（1319 行） | 749,738 | `3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14` |
| `gsm8k_train.jsonl` | 同上 `train.jsonl`（取前 5 条作 5-shot exemplar） | 4,166,206 | `17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465` |
| `mmlu_test.parquet` | HF `cais/mmlu` 的 `all/test`（14042 题，57 科，脚本每科取 5 题） | 3,504,718 | `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7` |

```bash
# GSM8K（任选一个镜像）
curl -sL -o gsm8k_test.jsonl  https://ghproxy.net/https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl
curl -sL -o gsm8k_train.jsonl https://ghproxy.net/https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl
# MMLU
curl -sL -o mmlu_test.parquet https://hf-mirror.com/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet
sha256sum gsm8k_test.jsonl gsm8k_train.jsonl mmlu_test.parquet   # 应与上表一致
```

跑之前把这三个文件和 `precision.py` 放在同一目录（或用 `--data` 指路径）：

```bash
source env.sh
python3 precision.py --dataset gsm8k --data data/gsm8k_test.jsonl  --train data/gsm8k_train.jsonl --shots 5 --conc 24 --out gsm8k_result.json
python3 precision.py --dataset mmlu  --data data/mmlu_test.parquet --per-subject 5 --conc 24 --out mmlu_result.json
```

> 说明：本实例（SCNet Notebook）无外网，上述数据是在本地下载后用 `tar | ssh` 传到
> `/root/hygon/eval/` 的；有外网的机器可直接在服务器上 curl。
