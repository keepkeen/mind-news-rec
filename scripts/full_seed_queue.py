"""Run the second full-BGE seed on each existing BGE machine after its first job."""
import argparse
import json
import time
from jobio import RUN,status,run_job
from fleet_queue import child

def main(mode):
    fit=mode=="refit"
    job="full_seed_"+mode+"_queue"
    predecessor="fullbge_checkpoint_replay" if fit else "queue_fullbge"
    status(job,"waiting_for_gpu",dependency=predecessor)
    while True:
        path=RUN/(predecessor+".status.json")
        if path.is_file() and json.loads(path.read_text())["phase"] in ("DONE","FAILED","SKIPPED"):break
        time.sleep(12)
    kind="fullbge_s2"
    args=["click_train.py",kind,"8192","192","64","3","1163"]
    name=kind+"_refit" if fit else kind
    status(job,"running",child=name)
    child(name,args+(["--refit"] if fit else []))
    if not fit:
        status(job,"running",child=kind+"_shadow")
        child(kind+"_shadow",args+["--shadow"])
    status(job,"DONE")
if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["evaluation","refit"])
    mode=parser.parse_args().mode
    run_job("full_seed_"+mode+"_queue",lambda:main(mode))
