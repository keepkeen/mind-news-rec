"""Launch an allocated recovery role without editing or starting any former session."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from colab_alerts import permitted
from colab_jobs import ART,JOBS,CHUNK,REMOTE,ROOT,atomic_json,call,deploy,rex,response
from recovery_watchdog import SESSIONS

def copy_refit(s,src,expected):
    if s not in ("mind_r7_bge","mind_r7_e5_fit_resume"):raise ValueError(s)
    name=src.name;count=(src.stat().st_size+CHUNK-1)//CHUNK
    prefix="restore_"+expected[:16]+"."
    code=f"""from pathlib import Path
import json
r=Path('/content/mind');out={{p.name:p.stat().st_size for p in (r/'incoming').glob({prefix!r}+'*')}}
print('@@MIND '+json.dumps(out))
"""
    existing=response(rex(s,code,55,2))
    with tempfile.TemporaryDirectory(dir=ART/"local",prefix="bge_upload_") as tmp:
        def send(i):
            chunk=prefix+f"{i:04d}";p=Path(tmp)/chunk
            if existing.get(chunk)==min(CHUNK,src.stat().st_size-i*CHUNK):return
            with src.open("rb") as f:f.seek(i*CHUNK);p.write_bytes(f.read(CHUNK))
            try:call(["upload","-s",s,str(p),REMOTE+"/incoming/"+chunk],100,3)
            finally:p.unlink(missing_ok=True)
        with ThreadPoolExecutor(max_workers=6) as pool:
            for done,_ in enumerate(pool.map(send,range(count)),1):
                if done%10==0 or done==count:
                    atomic_json(JOBS/s/"checkpoint_upload_progress.json",dict(name=name,done=done,total=count,utc=time.time()))
    code=f"""from pathlib import Path
import hashlib,json
r=Path('/content/mind');dest=r/'cache'/{name!r};tmp=dest.with_suffix('.uploading');parts=[r/'incoming'/({prefix!r}+'%04d'%i) for i in range({count})];h=hashlib.sha256()
with tmp.open('wb') as w:
 for p in parts:
  b=p.read_bytes();h.update(b);w.write(b)
assert h.hexdigest()=={expected!r};tmp.replace(dest);dest.with_suffix('.sha256').write_text(h.hexdigest())
for p in parts:p.unlink(missing_ok=True)
print('@@MIND '+json.dumps({{'sha256':h.hexdigest(),'bytes':dest.stat().st_size}}))
"""
    atomic_json(JOBS/s/"checkpoint_upload.json",response(rex(s,code,90,3)))

def main(s):
    d=JOBS/s
    while not (d/"machine.json").is_file():
        permitted()
        if (ART/"round7/no_new_assignments").is_file():return
        time.sleep(5)
    mode="evaluation" if s.endswith("e5_eval") else "refit" if s.endswith("e5_fit") else "bge"
    files=["jobio.py","ranking.py","recommender.py","click_text.py","click_train.py","mid_checkpoint.py","fleet_queue.py"]
    deploy(s,files+(["recovery_bge.py"] if mode=="bge" else ["e5_queue.py"]))
    if mode=="bge":
        prior=JOBS/"mind_fullfit_1119";receipt=prior/"fullbge_s2_refit_e2.receipt.json"
        meta=json.loads(receipt.read_text());identity=meta["checkpoint"];src=prior/identity["name"];expected=identity["sha256"]
        with src.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=expected:raise ValueError("Local refit checkpoint changed")
        dest=d/src.name
        if not dest.exists():os.link(src,dest)
        dest.with_name(dest.name+".verified").write_text(expected)
        call(["upload","-s",s,str(receipt),REMOTE+"/run/original_refit_receipt.json"],65,2)
        job=["bge_recovery_queue","recovery_bge.py"]
    else:job=["e5_"+mode+"_queue","e5_queue.py",mode,"--micro","256","--checkpoint-every","100"]
    subprocess.run([sys.executable,"scripts/colab_jobs.py","start","-s",s,"--",*job],cwd=ROOT,check=True)
    atomic_json(d/"dispatch.json",dict(utc=time.time(),command=job))
    if mode=="bge":copy_refit(s,src,expected)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("session",choices=SESSIONS);main(p.parse_args().session)
