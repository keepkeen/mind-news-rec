"""Run the candidate-only pilot on the first free seed machines; reuse the timed model if admitted."""
import json
from pathlib import Path
import sys
import time
from jobio import RUN,CACHE,status,run_job
from fleet_queue import child

def main(mode,queue):
    after="fleet_seedeval" if mode=="evaluation" else "fleet_seedfit"
    status(queue,"waiting_for_seeds",dependency=after)
    flags=[] if mode=="evaluation" else ["--refit"]
    suffix="" if mode=="evaluation" else "_refit"
    job="asym4"+suffix
    while True:
        p=RUN/(after+".status.json")
        if p.is_file() and json.loads(p.read_text()).get("phase") in ("DONE","FAILED"):break
        time.sleep(12)
    limit=time.time()+300
    while not (CACHE/"qwen.sha256").exists() and time.time()<limit:time.sleep(6)
    if not (CACHE/"qwen.sha256").exists():
        status(queue,"SKIPPED",reason="verified frozen cache unavailable");return
    admission=RUN/"asym_admission.json"
    if admission.exists() and not json.loads(admission.read_text())["run"]:
        status(queue,"SKIPPED",reason=json.loads(admission.read_text())["reason"]);return
    args=["click_train.py","asym4","32768","64","64","1","1144"]
    status(queue,"running_pilot_and_conditional_training",child=job)
    child(job,args+flags)
    state=json.loads((RUN/(job+".status.json")).read_text())
    if state["phase"]=="DONE" and mode=="evaluation":
        status(queue,"running",child="asym4_shadow");child("asym4_shadow",args+["--shadow"])
    status(queue,"DONE" if state["phase"]=="DONE" else "SKIPPED",outcome=state["phase"])
if __name__=="__main__":
    mode=sys.argv[1]
    if mode not in ("evaluation","refit"):raise ValueError(mode)
    queue="asym_"+mode+"_queue";run_job(queue,lambda:main(mode,queue))
