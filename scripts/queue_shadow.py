"""Produce past-only logits for honest temporal stacking of every supervised branch."""
import subprocess
import sys
from jobio import ROOT,RUN,status,run_job
from queue_branch import after

def main(kind):
    after(kind)
    cfg={"upper":["192","0.0005","1124"],"qwen":["512","0.0006","1123"]}[kind]
    name=kind+"_shadow"
    with (RUN/(name+".log")).open("ab",buffering=0) as log:
        p=subprocess.Popen(["nohup",sys.executable,"-u","recommender.py",kind,"2",*cfg,"--shadow"],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
    (RUN/(name+".pid")).write_text(str(p.pid))
    status("queue_"+name,"running",pid=p.pid)
    if p.wait()!=0:raise RuntimeError(name+" failed")
    status("queue_"+name,"DONE")
if __name__=="__main__":
    kind=sys.argv[1];run_job("queue_"+kind+"_shadow",lambda:main(kind))

