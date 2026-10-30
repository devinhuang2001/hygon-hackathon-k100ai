#!/usr/bin/env python3
"""AuK 零样本 TTS 批量跑 T04 条件清单（120 条 clone）。
输出: outputs/auk/<id>.wav （24kHz 单声道）
"""
import json, os, sys, time, traceback

BASE = os.path.dirname(os.path.abspath(__file__))
COND = os.path.join(BASE, "formal", "auk.jsonl")
OUTDIR = os.path.join(BASE, "evidence", "formal_auk", "outputs")
CKPT = "/root/hygon/models/auk/auk_base.safetensors"
QWEN = "/root/hygon/models/qwen-omni"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = [json.loads(l) for l in open(COND, encoding="utf-8") if l.strip()]
    todo = [r for r in rows if not os.path.exists(os.path.join(OUTDIR, r["id"] + ".wav"))]
    print(f"[auk] total={len(rows)} todo={len(todo)}", flush=True)
    if not todo:
        return
    sys.path.insert(0, "/root/hygon/work/AuK/src")
    # torchaudio 2.10 需要 TorchCodec；用 ffmpeg 打补丁替代 load/save
    import subprocess as _sp
    import torch as _torch
    import torchaudio as _ta
    import numpy as _np

    def _ffmpeg_load(path, *a, **k):
        probe = _sp.run(["ffprobe", "-v", "quiet", "-show_entries",
                         "stream=sample_rate,channels", "-of", "json", str(path)],
                        capture_output=True, check=True)
        info = json.loads(probe.stdout)["streams"][0]
        sr, ch = int(info["sample_rate"]), int(info.get("channels", 1))
        raw = _sp.run(["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "f32le", "-"],
                      capture_output=True, check=True).stdout
        audio = _np.frombuffer(raw, dtype=_np.float32)
        if ch > 1:
            audio = audio.reshape(-1, ch).mean(axis=1)
        return (_torch.from_numpy(audio.copy()).unsqueeze(0), sr)  # [1, T]

    def _ffmpeg_save(uri, source=None, sample_rate=None, *a, **k):
        # AuK 调用: torchaudio.save(output_path, audio_f32, sample_rate)
        path, tensor, sr = uri, source, sample_rate
        arr = tensor.detach().cpu().numpy()
        if arr.ndim > 1:
            arr = arr.squeeze()
        fmt = "f32le" if arr.dtype == _np.float32 else "s16le"
        _sp.run(["ffmpeg", "-y", "-v", "quiet", "-f", fmt, "-ar", str(int(sr)),
                 "-ac", "1", "-i", "-", "-ar", str(int(sr)), str(path)],
                input=arr.tobytes(), check=True)

    _ta.load = _ffmpeg_load
    _ta.save = _ffmpeg_save
    from auk.infer.infer_auk import AukInfer, save_audio
    engine = AukInfer(
        config_path="/root/hygon/models/auk/config.yaml",
        ckpt_path=CKPT, device="cuda", dtype="bf16",
        qwen_path=QWEN, cpu_offload=False)
    print("[auk] engine loaded", flush=True)
    for r in todo:
        rid = r["id"]
        outp = os.path.join(OUTDIR, rid + ".wav")
        try:
            # 官方 jsonl 的相对路径以 formal/conditions/ 为基准（../../data/...）
            ref = os.path.normpath(os.path.join(BASE, "formal", "conditions", r["assets"]["reference_audio"]["path"]))
            text = r["text"]
            p = r["params"]
            instruction = f"Say the following with the same voice: '{text}'"
            messages = [{"role": "user", "content": [
                {"type": "text", "text": instruction},
                {"type": "audio", "audio": ref},
            ]}]
            t0 = time.time()
            audio, sr = engine.generate(
                messages, audio=ref,
                gen_seconds=p["gen_seconds"],
                nfe=p["nfe"],
                cfg_strength=p["cfg_strength"],
                sway_sampling_coef=p["sway_sampling_coef"],
                seed=r["seed"])
            save_audio(audio, sr, outp)
            dur = audio.shape[-1] / sr
            exp = r.get("expected", {})
            lo, hi = exp.get("duration_min_s"), exp.get("duration_max_s")
            flag = "" if (lo is None or (lo <= dur <= hi)) else " [DURATION OUT OF RANGE]"
            print(f"[auk] OK {rid} dur={dur:.2f}s ({time.time()-t0:.0f}s){flag}", flush=True)
        except Exception as e:
            print(f"[auk] FAIL {rid}: {e}", flush=True)
            traceback.print_exc()
            with open(outp + ".error.txt", "w", encoding="utf-8") as f:
                f.write(str(e))


if __name__ == "__main__":
    main()
