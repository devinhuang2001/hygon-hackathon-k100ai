#!/usr/bin/env python3
"""Qwen3-ASR-1.7B 批量识别 T04 条件清单（客户端，打本地 vLLM /v1/audio/transcriptions）。
输出: outputs/qwen3_asr/<id>.txt （UTF-8 识别文本）
"""
import json, os, sys, time, threading

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # 仓库根（本脚本在 evidence/formal_asr/ 下）
COND = os.path.join(BASE, "formal", "conditions", "qwen3_asr.jsonl")
OUTDIR = os.path.join(BASE, "evidence", "formal_asr", "outputs")
PORT = os.environ.get("ASR_PORT", "8000")
SERVED = "qwen3-asr-1.7b"
CONC = int(os.environ.get("ASR_CONC", "8"))


def one(row, sem, lock, stats):
    rid = row["id"]
    outp = os.path.join(OUTDIR, rid + ".txt")
    if os.path.exists(outp) and os.path.getsize(outp) > 0:
        with lock:
            stats["skip"] += 1
        return
    audio_path = os.path.normpath(os.path.join(os.path.dirname(COND), row["assets"]["audio"]["path"]))
    # 服务端 PCM wav/flac 解码异常，统一转 mp3 后送（实测 mp3 路径转写正确）
    tmp_mp3 = "/tmp/asr_%s.mp3" % rid
    import subprocess
    subprocess.run(["ffmpeg", "-y", "-v", "quiet", "-i", audio_path,
                    "-ar", "16000", "-ac", "1", "-b:a", "128k", tmp_mp3], check=True)
    with sem:
        for attempt in range(3):
            try:
                import urllib.request, uuid
                boundary = uuid.uuid4().hex
                with open(tmp_mp3, "rb") as f:
                    audio_bytes = f.read()
                body = b""
                for name, val in (("model", SERVED.encode()),):
                    body += f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode() + val + b"\r\n"
                body += (f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; "
                         f"filename=\"{rid}.mp3\"\r\n"
                         "Content-Type: audio/mpeg\r\n\r\n").encode() + audio_bytes + b"\r\n"
                body += f"--{boundary}--\r\n".encode()
                req = urllib.request.Request(
                    f"http://127.0.0.1:{PORT}/v1/audio/transcriptions", data=body,
                    headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
                resp = json.loads(urllib.request.urlopen(req, timeout=300).read())
                text = resp.get("text", "")
                if not text:
                    raise RuntimeError("empty text: " + str(resp)[:200])
                with open(outp, "w", encoding="utf-8") as f:
                    f.write(text)
                with lock:
                    stats["ok"] += 1
                print(f"[asr] OK {rid} ({stats['ok']}/{stats['total']})", flush=True)
                return
            except Exception as e:
                err = str(e)[:200]
                time.sleep(2 + attempt * 3)
        with open(outp + ".error.txt", "w", encoding="utf-8") as f:
            f.write(err)
        with lock:
            stats["fail"] += 1
        print(f"[asr] FAIL {rid}: {err}", flush=True)
    try:
        os.unlink(tmp_mp3)
    except OSError:
        pass


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = [json.loads(l) for l in open(COND, encoding="utf-8") if l.strip()]
    stats = {"ok": 0, "fail": 0, "skip": 0, "total": len(rows)}
    sem, lock = threading.Semaphore(CONC), threading.Lock()
    t0 = time.time()
    ts = [threading.Thread(target=one, args=(r, sem, lock, stats)) for r in rows]
    [t.start() for t in ts]; [t.join() for t in ts]
    print(f"[asr] done in {time.time()-t0:.0f}s ok={stats['ok']} fail={stats['fail']} skip={stats['skip']}", flush=True)


if __name__ == "__main__":
    main()
