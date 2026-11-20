"""Run an assigned project role; each stage has its own PID, logfile, progress and epoch artifacts."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from jobio import ROOT,RUN,DATA,CACHE,EXPORT,run_job,status,wait_for

SEEDS=[("upper_s2",1141),("upper_s3",1142)]
def child(name,args):
    old=RUN/(name+".status.json")
    if old.exists() and json.loads(old.read_text()).get("phase")=="DONE":return
    with (RUN/(name+".log")).open("ab",buffering=0) as f:
        env=dict(os.environ)
        if "--host-upper-cache" in args:env["PYTORCH_ALLOC_CONF"]="expandable_segments:True"
        p=subprocess.Popen(["nohup",sys.executable,"-u",*args],cwd=ROOT,stdin=subprocess.DEVNULL,
                           stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=env)
    (RUN/(name+".pid")).write_text(str(p.pid))
    if p.wait():raise RuntimeError(name+" failed; see its log")
def main(a):
    fleet="fleet_"+a.role
    status(fleet,"deployment",mode=a.role)
    for g in ("shared","train","dev"):wait_for(DATA/(g+".ready"))
    if a.after:
        limit=time.time()+14*3600
        while time.time()<limit:
            p=RUN/(a.after+".status.json")
            if p.is_file() and json.loads(p.read_text()).get("phase") in ("DONE","FAILED","SKIPPED"):break
            time.sleep(8)
        else:raise TimeoutError(a.after)
    def run(name,cmd):
        status(fleet,"running",child=name);child(name,cmd)
    if a.role in ("fullfit","qfit"):
        cfg=["fullbge","8192","192","64","3","1125"] if a.role=="fullfit" else ["qlora","32768","160","64","2","1126"]
        run(cfg[0]+"_refit",["click_train.py",*cfg,"--refit"])
    if a.role=="frozenfit":
        for kind,batch,lr,seed in [("bge","768","0.0007","1119"),("qwen","512","0.0006","1121")]:
            wait_for(CACHE/(kind+".npy"),timeout=2*3600)
            run(kind+"_refit",["recommender.py",kind,"2",batch,lr,seed,"--refit"])
    if a.role=="upperfit":
        run("upper_refit",["recommender.py","upper","2","192","0.0005","1122","--refit",*(["--host-upper-cache"] if a.host else [])])
    if a.role in ("seedeval","seedfit"):
        def command(series,seed,epochs):
            return ["recommender.py","upper",str(epochs),"192","0.0005",str(seed),"--series",series,*(["--host-upper-cache"] if a.host else [])]
        for series,seed in SEEDS:
            refit=a.role=="seedfit"
            run(series+("_refit" if refit else ""),command(series,seed,2)+(["--refit"] if refit else []))
        if a.role=="seedeval":
            for series,seed in SEEDS:
                epoch=json.loads((EXPORT/(series+"_best.json")).read_text())["best_epoch"]
                run(series+"_shadow",command(series,seed,epoch)+["--shadow"])
    status(fleet,"DONE")
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("role",choices=["fullfit","qfit","frozenfit","upperfit","seedeval","seedfit"])
    p.add_argument("--after");p.add_argument("--host",action="store_true");a=p.parse_args()
    run_job("fleet_"+a.role,lambda:main(a))

