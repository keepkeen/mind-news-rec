"""Serialize the timed click experiment and its strictly earlier shadow on one named VM."""
import json
import subprocess
import sys
import time
from jobio import RUN,ROOT,status,run_job

def record(name):
    p=RUN/(name+".status.json")
    return json.loads(p.read_text()) if p.exists() else {}

def execute(name,command):
    with (RUN/(name+".log")).open("ab",buffering=0) as log:
        proc=subprocess.Popen(["nohup",sys.executable,"-u",*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                              stdin=subprocess.DEVNULL,start_new_session=True)
    (RUN/(name+".pid")).write_text(str(proc.pid))
    if proc.wait():raise RuntimeError("Training failed: "+name)

def main(kind,wide=False):
    dep={"fullbge":"qwen_shadow","qlora":"upper_shadow"}[kind]
    queue="queue_"+kind
    status(queue,"waiting_prior_job",dependency=dep)
    while record(dep).get("phase") not in ("DONE","FAILED"):time.sleep(8)
    params={"fullbge":["8192","192","64","3","1125"],"qlora":["16384","64","64","2","1126"]}[kind]
    if wide:
        if kind!="qlora":raise ValueError("Wide smoke is for the LoRA branch")
        params[:2]=["32768","160"]
    status(queue,"running",child=kind)
    execute(kind,["click_train.py",kind,*params])
    if record(kind).get("phase")=="SKIPPED":
        status(queue,"DONE",reason="six-hour smoke gate");return
    status(queue,"running",child=kind+"_shadow")
    execute(kind+"_shadow",["click_train.py",kind,*params,"--shadow"])
    status(queue,"DONE")
if __name__=="__main__":
    kind=sys.argv[1];run_job("queue_"+kind,lambda:main(kind,"--wide" in sys.argv[2:]))

