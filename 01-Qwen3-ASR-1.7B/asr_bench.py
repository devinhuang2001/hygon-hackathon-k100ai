import json,subprocess,threading,time,os
FILES=["data/zh.mp3","data/en.mp3","data/asr_example.wav"]
DUR={"zh.mp3":6.0,"en.mp3":8.0,"asr_example.wav":6.0}
def one(fn):
    t=time.time()
    r=subprocess.run(["curl","-s","--noproxy","*","-X","POST","http://127.0.0.1:8000/v1/audio/transcriptions",
        "-F",f"file=@{fn}","-F","model=qwen3-asr-1.7b","--max-time","300"],capture_output=True,text=True)
    el=time.time()-t
    try: ok = "text" in json.loads(r.stdout)
    except Exception: ok=False
    return el,ok,os.path.basename(fn)
for conc in (1,8,32):
    waves=max(1,64//(conc*len(FILES)))
    jobs=[(f,w) for w in range(waves) for f in FILES]*1
    jobs=jobs*(conc if conc==1 else 1)
    reqs=[FILES[i%3] for i in range(conc*waves*len(FILES)//len(FILES))]
    # build exact request list: conc in flight, total = conc*waves*3
    total=conc*waves*3
    reqs=[FILES[i%3] for i in range(total)]
    sem=threading.Semaphore(conc); res=[]
    lock=threading.Lock()
    def work(fn):
        with sem:
            r=one(fn)
            with lock: res.append(r)
    t0=time.time()
    ts=[threading.Thread(target=work,args=(f,)) for f in reqs]
    [t.start() for t in ts]; [t.join() for t in ts]
    wall=time.time()-t0
    audio=sum(DUR[os.path.basename(f)] for f in reqs)
    oks=sum(1 for r in res if r[1])
    print(f"conc={conc:3d} reqs={total:4d} ok={oks:4d} wall={wall:7.2f}s audio={audio:7.1f}s RTF={wall/audio:6.3f}x  speed={audio/wall:6.2f} audio-s/s  qps={total/wall:5.2f}")
