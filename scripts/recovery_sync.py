"""Immediately prioritize epoch checkpoints, keep one exact rolling checkpoint per live recovery VM."""
from concurrent.futures import ThreadPoolExecutor,as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import threading
import time
from colab_alerts import LATCH,AuthenticationBlocked,emit
from colab_jobs import ART,CHUNK,JOBS,REMOTE,atomic_json,call,completed_eval_epochs,heartbeat,omitted_test_cache,rex,response

ROLL=re.compile(r"^(?:e5(?:_refit|_shadow)?|fullbge_s2_shadow)_rolling_e\d+_s\d{4,}\.pt$")
EPOCH=re.compile(r"^(?:e5(?:_refit|_shadow)?|fullbge_s2_shadow)_e\d+\.pt$")
SCORES=re.compile(r"^(?:e5(?:_refit|_shadow|_trainonly)?|fullbge_s2_(?:refit|shadow))_e\d+_(?:dev|train|test)\.npy$")
# Leave room for concurrently arriving signed weights, score arrays and a test-scoring workspace.
OUTPUT_RESERVE=3*1024**3
def read(p):
    return json.loads(p.read_text()) if p.is_file() else {}
def obsolete(d,name):
    return bool(ROLL.fullmatch(name)) and name in read(d/"rolling_superseded.json")
def download_one(s,d,name,size,stamp,tmp):
    if Path(name).name!=name:raise ValueError(name)
    call(["download","-s",s,REMOTE+"/export/"+name,str(tmp)],135,3)
    if tmp.stat().st_size!=size:raise RuntimeError("Incomplete transfer "+name)

def assemble(s,d,meta,entries):
    started=time.time()
    name=meta["name"];target=d/name;marker=d/(name+".verified");sha=meta["sha256"]
    if target.is_file() and marker.is_file() and marker.read_text().strip()==sha:return
    parts=meta["parts"]
    if not all(p in entries for p in parts):raise RuntimeError("Waiting for complete remote parts: "+name)
    if any(p!=name+f".part{i:04d}" for i,p in enumerate(parts)) or Path(name).name!=name:raise ValueError(name)
    sizes=[entries[p][0] for p in parts];total=sum(sizes)
    tmp=d/("."+name+".receiving");track=d/("."+name+".transfer.json")
    state=read(track);done=set(state.get("complete",[])) if tmp.is_file() and state.get("sha256")==sha else set()
    if not done and shutil.disk_usage(d).free<total+128*1024**2:
        emit("disk",s,"本机可用空间不足以接收 "+name);raise RuntimeError("Insufficient disk: "+name)
    fd=os.open(tmp,os.O_RDWR|os.O_CREAT,0o600)
    try:
        os.ftruncate(fd,total)
        offsets=[];at=0
        for size in sizes:offsets.append(at);at+=size
        def get(i):
            p=parts[i];short=d/("."+p+".download")
            try:
                flags=() if EPOCH.fullmatch(name) else ("epoch_sync_active.json",)
                if ROLL.fullmatch(name):flags+=("scores_sync_active.json",)
                while any((d/f).exists() for f in flags) and not LATCH.exists():time.sleep(.5)
                download_one(s,d,p,sizes[i],entries[p][1],short)
                data=memoryview(short.read_bytes());offset=offsets[i]
                while data:
                    nw=os.pwrite(fd,data,offset)
                    if nw<=0:raise OSError("Short write")
                    data=data[nw:];offset+=nw
            finally:short.unlink(missing_ok=True)
            return i
        with ThreadPoolExecutor(max_workers=5) as pool:
            tasks=[pool.submit(get,i) for i in range(len(parts)) if i not in done]
            for f in as_completed(tasks):
                done.add(f.result())
                atomic_json(track,dict(sha256=sha,complete=sorted(done)))
        os.fsync(fd)
    finally:os.close(fd)
    with tmp.open("rb") as f:digest=hashlib.file_digest(f,"sha256").hexdigest()
    if digest!=sha:
        track.unlink(missing_ok=True);tmp.unlink(missing_ok=True);raise ValueError("SHA mismatch "+name)
    os.replace(tmp,target);mark=marker.with_suffix(".marking");mark.write_text(digest);mark.replace(marker)
    track.unlink(missing_ok=True)
    if name.endswith(".pt"):
        with (d/"checkpoint_receipts.lock").open("a") as lock:
            fcntl.flock(lock,fcntl.LOCK_EX)
            p=d/"checkpoint_transfers.json";log=read(p)
            log[name]=dict(transfer_started_utc=started,verified_utc=time.time(),remote_ready_utc=entries[name+".parts.json"][1]/1e9,sha256=sha,bytes=total)
            atomic_json(p,log)

def sync_priority(s,d,epoch):
    pattern=EPOCH if epoch else SCORES
    flag=d/("epoch_sync_active.json" if epoch else "scores_sync_active.json");flag.unlink(missing_ok=True)
    code=f"""from pathlib import Path
import json,re
r=Path('/content/mind/export');out=[]
for p in r.glob({'*_e*.pt.parts.json' if epoch else '*_e*.npy.parts.json'!r}):
 if not re.fullmatch({pattern.pattern!r},p.name.removesuffix('.parts.json')):continue
 v=json.loads(p.read_text());out.append([v,p.stat().st_mtime_ns,[(r/part).stat().st_size for part in v['parts']]])
print('@@MIND '+json.dumps(out))
"""
    while not (d/"stop_monitor").exists() and not LATCH.exists():
        try:
            selected=completed_eval_epochs() if not epoch else {}
            for m,stamp,sizes in response(rex(s,code,45,2)):
                n=m["name"];mark=d/(n+".verified")
                if not pattern.fullmatch(n):raise ValueError(n)
                if not epoch and omitted_test_cache(n,selected):continue
                if (d/n).is_file() and mark.is_file() and mark.read_text().strip()==m["sha256"]:continue
                atomic_json(flag,dict(name=n,started_utc=time.time(),remote_ready_utc=stamp/1e9))
                try:
                    entries={part:(size,stamp) for part,size in zip(m["parts"],sizes,strict=True)}
                    entries[n+".parts.json"]=(0,stamp)
                    atomic_json(d/(n+".parts.json"),m);assemble(s,d,m,entries)
                finally:flag.unlink(missing_ok=True)
        except AuthenticationBlocked:return
        except (OSError,RuntimeError,ValueError,KeyError) as exc:print(time.ctime(),flag.stem,str(exc)[-220:],flush=True)
        time.sleep(8)

def compact_rolling(s,d,meta,received,remote_entries):
    name=meta["name"]
    if not ROLL.fullmatch(name):return
    previous=read(d/"rolling_latest.json");old=read(d/"rolling_superseded.json")
    if previous and previous["name"]!=name:old[previous["name"]]=previous["sha256"]
    for n,(size,stamp) in remote_entries.items():
        if n.endswith(".parts.json") and ROLL.fullmatch(n[:-11]) and n[:-11]!=name and stamp<=received:
            old.setdefault(n[:-11],"older locally covered interval")
    atomic_json(d/"rolling_latest.json",dict(name=name,sha256=meta["sha256"],verified_utc=time.time(),remote_utc_ns=received))
    atomic_json(d/"rolling_superseded.json",old)
    for n in old:
        if not ROLL.fullmatch(n):raise ValueError(n)
        for suffix in ("",".verified",".parts.json"):(d/(n+suffix)).unlink(missing_ok=True)
    # A forgotten remote acknowledgement is recoverable from the durable local supersession record.
    garbage=[n for n in old if n+".parts.json" in remote_entries]
    if garbage:
        code=f"""from pathlib import Path
import json
r=Path('/content/mind/export')
for n in {garbage!r}:
 m=r/(n+'.parts.json')
 if not m.exists():continue
 for part in json.loads(m.read_text())['parts']:
  assert part.startswith(n+'.part') and Path(part).name==part
  (r/part).unlink(missing_ok=True)
 m.unlink()
print('@@MIND '+json.dumps({{'superseded':{len(garbage)}}}))
"""
        response(rex(s,code,50,2))

def sync(s):
    d=JOBS/s;d.mkdir(parents=True,exist_ok=True)
    threading.Thread(target=heartbeat,args=(s,),daemon=True).start()
    for epoch in (True,False):threading.Thread(target=sync_priority,args=(s,d,epoch),daemon=True).start()
    ledger=d/"sync.json";known=read(ledger)
    code="""import json,pathlib
r=pathlib.Path('/content/mind/export')
print('@@MIND '+json.dumps([[p.name,p.stat().st_size,p.stat().st_mtime_ns] for p in r.iterdir() if p.is_file() and not p.name.endswith('.tmp')]))
"""
    while not (d/"stop_monitor").exists() and not LATCH.exists():
        try:
            rows=response(rex(s,code,55,2));entries={n:(b,mt) for n,b,mt in rows}
            selected=completed_eval_epochs();old=read(d/"rolling_superseded.json")
            roll=[n for n in entries if n.endswith(".parts.json") and ROLL.fullmatch(n[:-11]) and n[:-11] not in old]
            newest=max(roll,key=lambda n:entries[n][1]) if roll else None
            small=[];large=[]
            for n,(size,stamp) in entries.items():
                if re.search(r"\.part\d+$",n) or omitted_test_cache(n,selected):continue
                if n.endswith(".parts.json"):
                    key=n[:-11]
                    if key in old or (ROLL.fullmatch(key) and n!=newest):continue
                    large.append(n)
                if known.get(n)!=[size,stamp] or not (d/n).is_file():small.append(n)
            def small_file(n):
                size,stamp=entries[n];tmp=d/("."+n+".small")
                download_one(s,d,n,size,stamp,tmp);os.replace(tmp,d/n);return n,[size,stamp]
            with ThreadPoolExecutor(max_workers=4) as pool:
                for n,stamp in pool.map(small_file,small):known[n]=stamp
            if small:atomic_json(ledger,known)
            # Epoch weights first, newest restart state next, candidate scores last.
            def priority(n):
                key=n[:-11];k=0 if key.endswith(".pt") and not ROLL.fullmatch(key) else 1 if ROLL.fullmatch(key) else 2
                return k,-entries[n][1]
            transferred=False
            for n in sorted(large,key=priority):
                m=read(d/n);key=m["name"];mark=d/(key+".verified")
                if EPOCH.fullmatch(key) or SCORES.fullmatch(key):continue
                if mark.is_file() and (d/key).is_file() and mark.read_text().strip()==m["sha256"]:
                    if ROLL.fullmatch(key) and read(d/"rolling_latest.json").get("name")!=key:compact_rolling(s,d,m,entries[n][1],entries)
                    continue
                if ROLL.fullmatch(key) and read(d/"rolling_latest.json") and not (d/("."+key+".receiving")).exists():
                    if shutil.disk_usage(d).free<sum(entries[p][0] for p in m["parts"])+OUTPUT_RESERVE:continue
                assemble(s,d,m,entries);compact_rolling(s,d,m,entries[n][1],entries);transferred=True;break
            if transferred:continue
        except AuthenticationBlocked:return
        except (OSError,RuntimeError,ValueError,KeyError) as exc:print(time.ctime(),"sync",str(exc)[-220:],flush=True)
        time.sleep(10)
