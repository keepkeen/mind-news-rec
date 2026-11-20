"""Deploy one of the two project E5 roles after reconciling its accelerator assignment."""
import argparse
import subprocess
import sys
import time
from colab_jobs import ART,JOBS,NAMES,ROOT,atomic_json,deploy

def main(s,mode,after=None):
    if s not in NAMES:raise ValueError(s)
    d=JOBS/s;status=ART/"round6"/("launch_"+mode+".json")
    limit=time.time()+720
    while not (d/"machine.json").is_file() and time.time()<limit:
        atomic_json(status,dict(stage="waiting_for_assignment",session=s,mode=mode))
        time.sleep(8)
    if not (d/"machine.json").is_file() or (d/"released.json").is_file():raise RuntimeError("No live assigned GPU")
    names=["jobio.py","ranking.py","recommender.py","click_text.py","click_train.py","fleet_queue.py","e5_queue.py"]
    deploy(s,names)
    args=["e5_"+mode+"_queue","e5_queue.py",mode,"--micro","256"]
    if after:args+=["--after",after]
    subprocess.run([sys.executable,"scripts/colab_jobs.py","start","-s",s,"--",*args],cwd=ROOT,check=True)
    atomic_json(status,dict(stage="remote_queue_started",session=s,mode=mode,dependency=after,utc=time.time()))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("session",choices=NAMES);p.add_argument("mode",choices=["evaluation","refit"]);p.add_argument("--after")
    a=p.parse_args();main(a.session,a.mode,a.after)
