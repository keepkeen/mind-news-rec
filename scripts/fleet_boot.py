"""Allocate one whitelisted MIND runtime, immediately start local monitoring and upload compact data."""
import fcntl
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import uuid
from colab_jobs import ART,CLI,JOBS,NAMES,ROOT,atomic_json,rex,response,upload_group
from colab_alerts import check as check_auth,permitted

def boot(session,gpu="A100",tries=3,shape=None):
    if session not in NAMES[2:]:raise ValueError(session)
    previous={"mind_e5_eval_1119":"mind_seed_eval_1119","mind_e5_fit_1119":"mind_seed_fit_1119"}.get(session)
    if previous and not (JOBS/previous/"released.json").is_file():raise RuntimeError("Predecessor still holds its fleet slot")
    d=JOBS/session;d.mkdir(exist_ok=True,parents=True)
    with (d/"allocation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        p=d/"allocation.json"
        rec=json.loads(p.read_text()) if p.exists() else dict(name=session,notebook=str(uuid.uuid4()),first_request=time.time(),gpu="A100")
        if gpu not in ("A100","H100"):raise ValueError(gpu)
        if not (d/"machine.json").exists():rec.update(gpu=gpu,high_ram=shape=="high")
        atomic_json(p,rec)
        if (d/"released.json").exists():return
        if not (d/"machine.json").exists():
            cli_python=Path(CLI).resolve().parent/"python3"
            for attempt in range(tries):
                permitted()
                if (ART/"fleet/no_allocation").exists():return
                atomic_json(d/"boot.json",dict(stage="requesting_"+rec["gpu"],attempt=attempt+1,time=time.time()))
                try:
                    job=subprocess.run([str(cli_python),str(ROOT/"scripts/assign_mind.py"),session,rec["notebook"],rec["gpu"],*([shape] if shape else [])],
                        cwd=ROOT,capture_output=True,text=True,timeout=180,env=dict(os.environ,COLAB_CLI_REQUEST_TIMEOUT="45",COLAB_CLI_ASSIGN_RETRIES="0"))
                    if job.returncode:check_auth(job.stdout+"\n"+job.stderr,session,"new")
                    if job.returncode and "Session READY" not in job.stdout:
                        reason=re.sub(r"https?://\S+","[endpoint]",job.stderr.strip())[-380:]
                        print("allocation response",reason,flush=True)
                except subprocess.TimeoutExpired as exc:
                    partial=" ".join(x.decode(errors="replace") if isinstance(x,bytes) else x or "" for x in (exc.stdout,exc.stderr))
                    check_auth(partial,session,"new")
                    print("assignment reply timed out; reconciling the same notebook",flush=True)
                try:
                    info=response(rex(session,"""
from pathlib import Path
import json,subprocess
root=Path('/content/mind')
for sub in ('incoming','compact','run','export','cache'): (root/sub).mkdir(parents=True,exist_ok=True)
print('@@MIND '+json.dumps({'gpu':subprocess.getoutput('nvidia-smi --query-gpu=name,memory.total --format=csv,noheader'),'uuid':subprocess.getoutput('nvidia-smi --query-gpu=uuid --format=csv,noheader'),'cwd':str(Path.cwd())}))
""",48,1))
                    if "A100" not in info["gpu"] and "H100" not in info["gpu"]:raise RuntimeError("Unexpected "+str(info))
                    info["allocated_utc"]=time.time();atomic_json(d/"machine.json",info);print(info,flush=True)
                    if (ART/"fleet/no_allocation").exists():
                        from project_budget import stop_one
                        stop_one(session);return
                    break
                except Exception as exc:
                    print("reconcile",str(exc)[-160:],flush=True)
                    if attempt+1==tries:
                        atomic_json(d/"boot.json",dict(stage="CAPACITY_UNAVAILABLE",time=time.time(),reason=str(exc)[-170:]))
                        return
                    time.sleep(20*(attempt+1))
        monitor="monitor_"+session
        pp=ART/"local"/(monitor+".pid")
        live=False
        if pp.exists():
            try:os.kill(int(pp.read_text()),0);live=True
            except ProcessLookupError:pass
        if not live:
            subprocess.run([sys.executable,"scripts/detach.py",monitor,"/usr/bin/caffeinate","-i",
                            sys.executable,"scripts/colab_jobs.py","monitor","-s",session],cwd=ROOT,check=True)
        for g in ("shared","train","dev","test"):
            for attempt in range(3):
                try:
                    atomic_json(d/"boot.json",dict(stage="upload",group=g,time=time.time()))
                    upload_group(session,g);break
                except RuntimeError:
                    if attempt==2:raise
                    time.sleep(5)
        atomic_json(d/"boot.json",dict(stage="READY",time=time.time()))
if __name__=="__main__":boot(sys.argv[1],sys.argv[2] if len(sys.argv)>2 else "A100",int(sys.argv[3]) if len(sys.argv)>3 else 3,sys.argv[4] if len(sys.argv)>4 else None)

