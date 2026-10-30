#!/usr/bin/env python3
# 正确性证据：按官方基线文件写死的配置调用服务跑固定子集，和官方 GPU 基线比；只罚劣化，不罚提升。
# 说明：T04 的 ASR 评测器（WER/CER）与官方基线文件随分题材料下发，本仓库未持有，
#       verify.py 保留官方判分逻辑与 --selfcheck；把官方 baselines/<模型>.json 与 data/ 放入后即可直接跑。
import os, sys, json, argparse

TOL = 0.02  # 统一容差：劣化不超过 2%；有满分的按满分计（百分制即 2 pt），否则按基线值计

def check(got, m):
    # 越低越好的指标（WER、AAS 等）在基线文件里标 lower_is_better
    worse = got - m["ref"] if m.get("lower_is_better") else m["ref"] - got
    tol = TOL * m.get("scale", m["ref"])
    return {"got": round(got, 2), "ref": m["ref"], "worse": round(worse, 2),
            "tol": round(tol, 2), "pass": worse <= tol + 1e-9}

def selfcheck():
    # 逻辑自检：劣化超容差必须 FAIL，容差内与提升必须 PASS
    acc = {"ref": 80.0, "scale": 100}
    wer = {"ref": 5.0, "scale": 100, "lower_is_better": True}
    assert not check(77.9, acc)["pass"], "掉 2.1 pt 必须 FAIL"
    assert check(78.0, acc)["pass"] and check(85.0, acc)["pass"], "掉 2 pt 以内或涨点必须 PASS"
    assert not check(7.1, wer)["pass"] and check(4.0, wer)["pass"], "WER 升超容差 FAIL，下降 PASS"
    print("selfcheck OK")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    args = ap.parse_args()
    if args.selfcheck:
        selfcheck(); sys.exit(0)

    model_id = os.environ.get("MODEL_ID", "")
    base_path = f"baselines/{model_id.replace('/', '__')}.json"
    if not os.path.exists(base_path):
        sys.exit(f"缺少官方基线文件 {base_path}（随分题材料下发，需向赛事方获取），"
                 f"无法产出 verify_result.json。判分逻辑可用 --selfcheck 验证。")
    with open(base_path, encoding="utf-8") as f:
        base = json.load(f)
    # 评测器 EVALUATORS（gsm8k/mmlu/wer/cer/aas…）与 data/ 子集同样随分题材料下发，
    # 拿到后在此注册即可；本文件不臆造评测实现。
    sys.exit(f"基线文件已就绪，但缺少与 {list(base.get('metrics', {}))} 对应的评测器与 data/ 子集"
             f"（随分题材料下发），请放入后重跑。")

if __name__ == "__main__":
    main()
