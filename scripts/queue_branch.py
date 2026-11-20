"""Execute dependent GPU tasks serially. Children have their own log, status and PID."""
import json
import os
import subprocess
import sys
import time
from jobio import ROOT,RUN,status,run_job

def after(job):
    p=RUN/(job+".status.json")
    while True:
        if p.exists():
            s=json.loads(p.read_text())
            if s["phase"]=="DONE":return
            if s["phase"]=="FAILED":raise RuntimeError(job+" failed: "+s.get("error",""))
        time.sleep(10)

def child(name,args):
    status("followup","running",child=name)
    with (RUN/(name+".log")).open("ab",buffering=0) as f:
        p=subprocess.Popen(["nohup",sys.executable,"-u",*args],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                           stdin=subprocess.DEVNULL,start_new_session=True)
    (RUN/(name+".pid")).write_text(str(p.pid))
    if p.wait()!=0:raise RuntimeError("Child failed: "+name)

def main():
    after("bge")
    child("bge_shadow",["recommender.py","bge","2","768","0.0007","1120","--shadow"])
    child("embed_qwen",["encode.py","qwen","Qwen/Qwen3-Embedding-8B","1d8ad4ca9b3dd8059ad90a75d4983776a23d44af","64","80"])
    child("qwen",["recommender.py","qwen","2","512","0.0006","1121"])
    if "--upper" in sys.argv:
        child("upper",["recommender.py","upper","2","192","0.0005","1122"])
    status("followup","DONE")
if __name__=="__main__":run_job("followup",main)

