"""Bounded Colab CLI calls, chunked checked transfer and independent heartbeat/sync."""
import argparse
import fcntl
import concurrent.futures as cf
import gzip
import hashlib
import io
import json
import os
import re
from pathlib import Path
import subprocess
import sys
import tarfile
import threading
import time
from colab_alerts import LATCH,AuthenticationBlocked,permitted,check as check_auth

ROOT=Path(__file__).resolve().parents[1]
ART=ROOT/"artifacts"
JOBS=ART/"jobs"
CLI=str(Path.home()/".local/bin/colab")
NAMES=("mind_h100","mind_plm_1119","mind_fullfit_1119","mind_qfit_1119","mind_seed_eval_1119","mind_seed_fit_1119",
       "mind_e5_eval_1119","mind_e5_fit_1119","mind_r7_e5_eval","mind_r7_e5_fit","mind_r7_bge","mind_r7_e5_fit_resume")
CHUNK=8*1024*1024
REMOTE="/content/mind"

def completed_eval_epochs():
    selected={}
    active_refits=set()
    for d in (JOBS/name for name in NAMES):
        hp=d/"heartbeat.json"
        if not hp.exists():continue
        states=json.loads(hp.read_text())["status"]
        if not (d/"released.json").exists() and (not (ART/"round7/incident_accounting.json").exists() or d.name.startswith("mind_r7_")):
            for k in ("fullbge_s2","e5"):
                phase=states.get(k+"_refit.status.json",{}).get("phase")
                if phase and phase not in ("DONE","FAILED","SKIPPED","MOVED"):active_refits.add(k)
        for k in ("bge","upper","qwen","upper_s2","upper_s3","fullbge","fullbge_s2","qlora","asym4","e5"):
            if states.get(k+".status.json",{}).get("phase")!="DONE":continue
            p=d/(k+"_best.json")
            if p.exists():selected[k]=json.loads(p.read_text())["best_epoch"]
    # While a fit is live, its test caches are safe on the VM. Download the selected epoch when known.
    # When a refit finishes first, sync all still-possible epochs before releasing that VM.
    for k in active_refits:selected.setdefault(k,0)
    if (ART/"round7/bge_epoch_choice.json").is_file():selected["fullbge_s2"]=2
    return selected

def omitted_test_cache(name,selected):
    # Every epoch checkpoint is synchronized. Optional caches are reproducible from those files.
    m=re.fullmatch(r"(bge|upper|qwen|upper_s[23]|fullbge(?:_s2)?|qlora|asym4|e5)(_refit)?_e(\d+)(_replay)?_test\.npy(?:\.part\d+|\.parts\.json)?",name)
    if not m:return False
    kind,refit,epoch,replay=m.groups()
    if not refit:return kind not in ("bge","upper","qwen")
    if kind=="e5" and (ART/"round7/e5_fit_incident.json").is_file():
        # This replacement can disappear before the eval curve finishes. Keep its epoch test scores as soon as they arrive.
        return False
    if kind in ("fullbge","qlora") and not replay:return True
    return kind in selected and int(epoch)!=selected[kind]

def atomic_json(p, value):
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def call(args,timeout=90,attempts=3,code=None):
    for attempt in range(attempts):
        permitted()
        try:
            p=subprocess.run([CLI,*args],input=code,text=True,capture_output=True,timeout=timeout)
            out=re.sub(r"(?i)(colab-runtime-proxy-token=)[^\s&]+",r"\1[redacted]",p.stdout+"\n"+p.stderr)
            if p.returncode or "authorization_response" in out:
                check_auth(out,args[args.index("-s")+1] if "-s" in args else "MIND",args[0])
            if p.returncode: raise RuntimeError(out[-900:])
            return out.strip()
        except (subprocess.TimeoutExpired,RuntimeError) as e:
            if isinstance(e,subprocess.TimeoutExpired):
                partial=" ".join(x.decode(errors="replace") if isinstance(x,bytes) else x or "" for x in (e.stdout,e.stderr))
                check_auth(partial,args[args.index("-s")+1] if "-s" in args else "MIND",args[0])
            if attempt+1==attempts: raise RuntimeError(f"{args[0]} failed: {str(e)[-950:]}") from e
            time.sleep(2**attempt)

def rex(session, code, timeout=60,attempts=3):
    d=JOBS/session
    d.mkdir(parents=True,exist_ok=True)
    with (d/"kernel.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return call(["exec","-s",session,"--timeout",str(timeout-10)],timeout,attempts,code)

def response(text):
    for line in reversed(text.splitlines()):
        if "@@MIND " in line: return json.loads(line.split("@@MIND ",1)[1])
    raise RuntimeError("No remote acknowledgement: "+text[-700:])

def provision(session):
    for gpu in ("A100",):
        try:
            out=call(["new","-s",session,"--gpu",gpu],320,1)
            print(out[-450:],flush=True)
        except RuntimeError as e: print(str(e)[-450:],flush=True)
        try:
            info=response(rex(session,'''
import json,os,subprocess
from pathlib import Path
for d in ("mind/incoming","mind/compact","mind/export","mind/run","mind/cache"):
    (Path("/content")/d).mkdir(exist_ok=True,parents=True)
print("@@MIND "+json.dumps({"cwd":os.getcwd(),"gpu":subprocess.getoutput("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader"),"free":subprocess.getoutput("df -h /content | tail -1")}))
''',60,1))
            if not any(x in info["gpu"] for x in ("H100","A100")):
                raise RuntimeError("Unexpected accelerator "+info["gpu"])
            atomic_json(JOBS/session/"machine.json",info); print(info,flush=True); return
        except RuntimeError as e: print(str(e)[-400:],flush=True)
    raise RuntimeError("A100 unavailable for "+session)

def chunks(group):
    base=ART/"compact"; target=ART/"upload"/group;target.mkdir(parents=True,exist_ok=True)
    archive=target/"data.tgz"
    items=sorted(p for p in base.iterdir() if p.is_file()) if group=="shared" else [base/group]
    with archive.open("wb") as f, gzip.GzipFile(fileobj=f,mode="wb",compresslevel=2,mtime=0) as gz, tarfile.open(fileobj=gz,mode="w|") as tar:
        for p in items: tar.add(p,arcname=p.name)
    files=[]
    with archive.open("rb") as f:
        while data:=f.read(CHUNK):
            name=f"{group}.{len(files):04d}"
            (target/name).write_bytes(data)
            files.append([name,len(data),hashlib.sha256(data).hexdigest()])
    archive.unlink()
    atomic_json(target/"manifest.json",files)
    print(group,sum(q[1] for q in files),len(files),flush=True)

def upload_group(session,group):
    target=ART/"upload"/group
    files=json.loads((target/"manifest.json").read_text())
    query=f"""
import pathlib,json
root=pathlib.Path("{REMOTE}")
print("@@MIND "+json.dumps({{"ready":(root/"compact/{group}.ready").exists(),
    "chunks":{{p.name:p.stat().st_size for p in (root/"incoming").glob("{group}.[0-9][0-9][0-9][0-9]")}}}}))
"""
    remote=response(rex(session,query,55,3))
    if remote["ready"]:
        print(group,"already verified on runtime",flush=True)
        return
    def send(f):
        name,size,sha=f
        call(["upload","-s",session,str(target/name),REMOTE+"/incoming/"+name],130,4)
    missing=[f for f in files if remote["chunks"].get(f[0])!=f[1]]
    with cf.ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(send,missing))
    unpack=f'''
import hashlib,json,tarfile,pathlib
r=pathlib.Path("{REMOTE}"); fs={files!r}
archive=r/"incoming/{group}.tgz"
with archive.open("wb") as out:
    for name,n,sha in fs:
        b=(r/"incoming"/name).read_bytes()
        assert len(b)==n and hashlib.sha256(b).hexdigest()==sha,name
        out.write(b)
with tarfile.open(archive,"r:gz") as tf: tf.extractall(r/"compact",filter="data")
(r/"compact/{group}.ready").touch()
archive.unlink()
for name,n,sha in fs: (r/"incoming"/name).unlink(missing_ok=True)
'''
    cmd="import json,pathlib\nif not pathlib.Path("+repr(REMOTE+"/compact/"+group+".ready")+").exists():\n"+ "".join("    "+l+"\n" for l in unpack.strip().splitlines())
    cmd+="print('@@MIND '+json.dumps("+repr({"group":group,"chunks":len(files)})+"))"
    print(response(rex(session,cmd,75,4)),flush=True)

def deploy(session, filenames):
    for name in filenames:
        p=ROOT/"scripts"/name
        call(["upload","-s",session,str(p),REMOTE+"/"+p.name],90)
    print("uploaded code",filenames,flush=True)

def heartbeat(session):
    d=JOBS/session;d.mkdir(parents=True,exist_ok=True)
    code='''
import json,time,pathlib
from pathlib import Path
r=Path("/content/mind")
status={f.name:json.loads(f.read_text()) for f in (r/"run").glob("*.status.json")}
for f in (r/"run").glob("*.log"):
    target=r/"export"/(f.name+".txt")
    if not target.exists() or target.stat().st_mtime < f.stat().st_mtime:
        with f.open("rb") as log:
            log.seek(max(0,f.stat().st_size-300000))
            tmp=target.with_suffix(".tmp");tmp.write_bytes(log.read());tmp.replace(target)
print("@@MIND "+json.dumps({"timestamp":time.time(),"status":status,"ready":[p.stem for p in (r/"compact").glob("*.ready")]}))
'''
    while True:
        if LATCH.exists():return
        start=time.monotonic()
        try: atomic_json(d/"heartbeat.json",response(rex(session,code,48,2)))
        except AuthenticationBlocked:return
        except Exception as e: print(time.ctime(),"heartbeat",str(e)[-500:],flush=True)
        if (d/"stop_monitor").exists(): return
        time.sleep(max(1,180-(time.monotonic()-start)))

def sync(session):
    dest=JOBS/session;dest.mkdir(parents=True,exist_ok=True)
    code='''
import json, pathlib
root=pathlib.Path("/content/mind/export")
print("@@MIND "+json.dumps([[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in root.iterdir() if p.is_file() and not p.name.endswith(".tmp")]))
'''
    # Heartbeat runs on its own clock even if a data transfer is slow.
    threading.Thread(target=heartbeat,args=(session,),daemon=True).start()
    ledger=dest/"sync.json"
    known=json.loads(ledger.read_text()) if ledger.exists() else {}
    while not (dest/"stop_monitor").exists():
        if LATCH.exists():return
        try:
            entries=response(rex(session,code,55,2))
            completed_parts=set()
            for manifest in dest.glob("*.parts.json"):
                info=json.loads(manifest.read_text())
                target=dest/info["name"];verified=dest/(info["name"]+".verified")
                if target.is_file() and verified.is_file() and verified.read_text().strip()==info["sha256"]:
                    completed_parts.update(info["parts"])
            for part in completed_parts:
                (dest/part).unlink(missing_ok=True)
            selected=completed_eval_epochs()
            for part in dest.glob("*.npy.part[0-9][0-9][0-9][0-9]"):
                if omitted_test_cache(part.name,selected):part.unlink(missing_ok=True)
            todo=[(n,b,mt) for n,b,mt in entries if n not in completed_parts and not omitted_test_cache(n,selected)
                  and (known.get(n)!=[b,mt] or not (dest/n).exists())]
            todo.sort(key=lambda e:0 if e[0].endswith((".json",".log.txt")) else 1)
            def get(e):
                n,b,mt=e
                if Path(n).name!=n: raise ValueError(n)
                tmp=dest/(n+".downloading")
                call(["download","-s",session,REMOTE+"/export/"+n,str(tmp)],130,3)
                if tmp.stat().st_size!=b: raise RuntimeError("Incomplete/changing file: "+n)
                tmp.replace(dest/n)
                return n,[b,mt]
            with cf.ThreadPoolExecutor(max_workers=4) as pool:
                for n,stamp in pool.map(get,todo): known[n]=stamp
            atomic_json(ledger,known)
            # Concatenate completed immutable outputs and verify their remote digest.
            for f in dest.glob("*.parts.json"):
                info=json.loads(f.read_text()); name=info["name"]; out=dest/name
                if out.exists() and (dest/(name+".verified")).is_file(): continue
                if not all((dest/p).is_file() for p in info["parts"]): continue
                tmp=dest/(name+".assembling"); h=hashlib.sha256()
                with tmp.open("wb") as w:
                    for p in info["parts"]:
                        data=(dest/p).read_bytes();w.write(data);h.update(data)
                if h.hexdigest()!=info["sha256"]: raise RuntimeError("Hash mismatch "+name)
                tmp.replace(out);(dest/(name+".verified")).write_text(h.hexdigest())
                for part in info["parts"]:(dest/part).unlink(missing_ok=True)
        except Exception as e: print(time.ctime(),"sync",str(e)[-500:],flush=True)
        time.sleep(55)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["new","chunks","upload","deploy","monitor","start","exec"])
    p.add_argument("-s","--session",choices=NAMES,default=NAMES[0])
    p.add_argument("items",nargs="*")
    a=p.parse_intermixed_args()
    if a.action=="chunks":
        for g in a.items: chunks(g)
    elif a.action=="new": provision(a.session)
    elif a.action=="upload":
        for g in a.items: upload_group(a.session,g)
    elif a.action=="deploy": deploy(a.session,a.items)
    elif a.action=="monitor":
        if a.session.startswith("mind_r7_"):
            from recovery_sync import sync as sync_recovery
            sync_recovery(a.session)
        else:sync(a.session)
    elif a.action=="exec": print(rex(a.session,sys.stdin.read()))
    else:
        name,*cmd=a.items
        if not name.replace("_","").isalnum(): raise ValueError(name)
        code=f'''
import os, subprocess, json, pathlib, time
root=pathlib.Path("{REMOTE}"); pidfile=root/"run/{name}.pid"
pid=int(pidfile.read_text()) if pidfile.exists() else 0
if not pid or not pathlib.Path("/proc/"+str(pid)).exists():
    log=open(root/"run/{name}.log","ab",buffering=0)
    env=dict(os.environ,PYTHONUNBUFFERED="1",TOKENIZERS_PARALLELISM="false",OMP_NUM_THREADS="3",OPENBLAS_NUM_THREADS="1")
    p=subprocess.Popen(["nohup","python","-u"]+{cmd!r},cwd=root,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,env=env)
    pid=p.pid;pidfile.write_text(str(pid))
print("@@MIND "+json.dumps({{"job":"{name}","pid":pid}}))
'''
        print(response(rex(a.session,code,65)))
if __name__=="__main__": main()

