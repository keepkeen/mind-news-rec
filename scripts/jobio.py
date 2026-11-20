"""Remote job progress and immutable chunk exports. Absolute paths belong to this job."""
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

ROOT=Path(os.environ.get("MIND_WORK",".")).resolve()
DATA=ROOT/"compact"
RUN=ROOT/"run"
EXPORT=ROOT/"export"
CACHE=ROOT/"cache"
for p in (RUN,EXPORT,CACHE): p.mkdir(parents=True,exist_ok=True)

def write_json(path,obj):
    tmp=path.with_suffix(path.suffix+".tmp")
    tmp.write_text(json.dumps(obj,indent=2,default=str))
    tmp.replace(path)

def status(job,phase,**data):
    obj=dict(job=job,phase=phase,utc_epoch=time.time(),**data)
    write_json(RUN/(job+".status.json"),obj)
    print(json.dumps(obj,default=str),flush=True)

def run_job(job,fn):
    try: fn()
    except BaseException as e:
        status(job,"FAILED",error=repr(e),traceback=traceback.format_exc()[-2600:])
        raise

def export_file(path):
    path=Path(path)
    if path.stat().st_size<=7*1024**2:
        tmp=EXPORT/(path.name+".tmp");tmp.write_bytes(path.read_bytes());tmp.replace(EXPORT/path.name)
        return
    h=hashlib.sha256()
    parts=[]
    with path.open("rb") as f:
        while b:=f.read(8*1024**2):
            n=path.name+f".part{len(parts):04d}"
            tmp=EXPORT/(n+".tmp");tmp.write_bytes(b);tmp.replace(EXPORT/n)
            h.update(b);parts.append(n)
    write_json(EXPORT/(path.name+".parts.json"),dict(name=path.name,parts=parts,sha256=h.hexdigest()))

def wait_for(path,timeout=10800):
    start=time.time()
    while not Path(path).exists():
        if time.time()-start>timeout: raise RuntimeError("Input not received: "+str(path))
        time.sleep(5)

def refit_receipt(series,epoch,seed,checkpoint,scores,npos,batch,checkpoint_scored=False):
    def identity(p):
        p=Path(p);manifest=EXPORT/(p.name+".parts.json")
        sha=json.loads(manifest.read_text())["sha256"] if manifest.exists() else hashlib.sha256(p.read_bytes()).hexdigest()
        return dict(name=p.name,bytes=p.stat().st_size,sha256=sha)
    write_json(EXPORT/f"{series}_refit_e{epoch}.receipt.json",
               dict(series=series,epoch=epoch,seed=seed,training=["train","dev"],positive_pairs=npos,batch=batch,
                    checkpoint=identity(checkpoint),test_scores=identity(scores),checkpoint_scored=checkpoint_scored,
                    selection="matching epoch of train-only dev curve"))

