"""Time both E5 fits before shared admission, then run train/dev or train+dev in the assigned slot."""
import argparse
import json
import time
from fleet_queue import child
from jobio import DATA,RUN,status,run_job,wait_for

def main(mode,batch,micro,after,checkpoint_every,resume=None):
    fit=mode=="refit";queue="e5_"+mode+"_queue"
    status(queue,"waiting_for_inputs",dependency=after)
    for name in ("shared","train","dev"):wait_for(DATA/(name+".ready"))
    if after:
        p=RUN/(after+".status.json")
        while not p.exists() or json.loads(p.read_text()).get("phase") not in ("DONE","FAILED","SKIPPED"):time.sleep(10)
    args=["click_train.py","e5",str(batch),str(micro),"64","3","1125"]
    if checkpoint_every:args+=["--checkpoint-every",str(checkpoint_every)]
    if resume:
        if not fit:raise ValueError("The resume queue is for the train+dev fit")
        args+=["--resume",resume]
    job="e5_refit" if fit else "e5"
    status(queue,"running",child=job)
    child(job,args+(["--refit"] if fit else []))
    outcome=json.loads((RUN/(job+".status.json")).read_text())["phase"]
    if not fit and outcome=="DONE":
        status(queue,"running",child="e5_shadow")
        child("e5_shadow",args+["--shadow"])
    status(queue,"DONE" if outcome=="DONE" else "SKIPPED",outcome=outcome)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("mode",choices=["evaluation","refit"]);p.add_argument("--batch",type=int,default=8192)
    p.add_argument("--micro",type=int,default=192);p.add_argument("--after")
    p.add_argument("--checkpoint-every",type=int,default=0);p.add_argument("--resume")
    a=p.parse_args();run_job("e5_"+a.mode+"_queue",lambda:main(a.mode,a.batch,a.micro,a.after,a.checkpoint_every,a.resume))
