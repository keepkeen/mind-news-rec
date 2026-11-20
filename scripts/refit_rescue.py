"""Recover only the MIND E5 refit and report deadline forecasts from completed optimizer steps."""
import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from zoneinfo import ZoneInfo
from colab_alerts import LATCH,permitted
from colab_jobs import ART,JOBS,ROOT,CLI,atomic_json,deploy
from fleet_boot import boot
from project_budget import ledger,STOP_AT,stop_one

OLD="mind_r7_e5_fit"
RESUME="mind_r7_e5_fit_resume"
EVAL="mind_r7_e5_eval"
WORK=ART/"round7"
TARGET=datetime(2026,11,20,4,15,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
LOCK=TARGET+900
def read(p):
    return json.loads(p.read_text()) if p.is_file() else {}

def allocate():
    d=JOBS/RESUME
    for attempt in range(100):
        permitted()
        if (d/"machine.json").is_file() or (WORK/"no_new_assignments").exists():return
        now=time.time()
        # A later start can still deliver e1/e2 if selected by the independent eval curve.
        if now>TARGET-1.2*3600 or ledger()["protected_cu"]>STOP_AT-12:return
        high=attempt%2==0
        atomic_json(d/"retry_state.json",dict(utc=now,attempt=attempt+1,high_ram=high))
        try:boot(RESUME,"A100",1,"high" if high else None)
        except BlockingIOError:pass
        if (d/"machine.json").exists():return
        time.sleep(min(65,10+attempt*8))

def start_queue(s,job,args):
    subprocess.run([sys.executable,"scripts/colab_jobs.py","start","-s",s,"--",job,*args],cwd=ROOT,check=True)

def dispatch():
    from e5_admission import deliver
    from recovery_launch import copy_refit
    d=JOBS/RESUME
    while not (d/"machine.json").is_file():
        permitted()
        if time.time()>TARGET-3600 or (WORK/"no_new_assignments").exists():return
        time.sleep(4)
    rolling=read(JOBS/OLD/"rolling_latest.json");src=JOBS/OLD/rolling["name"]
    signed=src.with_name(src.name+".verified").read_text().strip()
    with src.open("rb") as f:
        if hashlib.file_digest(f,"sha256").hexdigest()!=signed or signed!=rolling["sha256"]:raise ValueError("Invalid local resume")
    with ThreadPoolExecutor(max_workers=1) as pool:
        transfer=pool.submit(copy_refit,RESUME,src,signed) if not (d/"checkpoint_upload.json").exists() else None
        if not (d/"dispatch.json").exists():
            deploy(RESUME,["jobio.py","ranking.py","recommender.py","click_text.py","click_train.py","mid_checkpoint.py","fleet_queue.py","e5_queue.py"])
            book=ledger()
            # Reserve until the 04:30 lock for both active roles at their actual planning rates.
            rates=book["protected_rates"];left=max(0,LOCK-time.time())/3600
            spend=book["protected_cu"]+sum(left*rates[book["sessions"][s]["gpu"]] for s in (EVAL,RESUME) if book["sessions"][s]["state"]=="active")
            # Actual remaining work is shorter than a whole lock interval if the estimate approaches the stop guard.
            if spend>=STOP_AT:
                ev=read(WORK/"admission.json")["finish_utc"][EVAL]
                spend=book["protected_cu"]+max(0,ev-time.time())/3600*rates[book["sessions"][EVAL]["gpu"]]+left*rates[book["sessions"][RESUME]["gpu"]]
            if spend>=STOP_AT:
                stop_one(RESUME,"preserve CU for the independent eval/shadow and train-only fallback");return
            deliver(RESUME,dict(run=True,reason="verified optimizer resume; new machine runs its own timing and gradient-parity smoke",upper_cu=spend))
            args=["e5_queue.py","refit","--micro","256","--checkpoint-every","100","--resume","cache/"+src.name]
            start_queue(RESUME,"e5_refit_queue",args)
            atomic_json(d/"dispatch.json",dict(utc=time.time(),args=args,checkpoint=src.name,sha256=signed,upper_at_dispatch=spend))
        if transfer:transfer.result()

def states(s):
    d=JOBS/s
    out={v["job"]:v for v in read(d/"heartbeat.json").get("status",{}).values()}
    for k,v in read(d/"parallel_probe.json").get("states",{}).items():
        if v.get("utc_epoch",0)>out.get(k,{}).get("utc_epoch",0):out[k]=v
    return out

def estimate(now=None):
    now=time.time() if now is None else now
    book=ledger(now)
    ev=states(EVAL);ft=states(RESUME)
    main=ev.get("e5",{});shadow=ev.get("e5_shadow",{});fit=ft.get("e5_refit",{})
    step=int(read(WORK/"e5_fit_incident.json")["rolling"]["name"].split("_s")[-1].split(".")[0])
    rolled=read(JOBS/RESUME/"e5_refit_resumed.json")
    def residual(row,ep,step,count,target,rate):
        if row.get("phase")=="DONE":return 0.,0.
        current_ep=int(row.get("epoch",ep))
        current_step=int(row.get("step",step)) if row.get("phase")=="training" else (count if row.get("phase")=="refresh_for_evaluation" else step)
        rate=max(rate,float(row.get("sec_per_step",0))) if row.get("step",0)>=10 else rate
        left=max(0,(target-current_ep)*count+count-current_step)
        return left*rate,left*rate*1.10
    chosen=int(read(JOBS/EVAL/"e5_best.json").get("best_epoch",3)) if main.get("phase")=="DONE" else 3
    def pending_weights(ep):
        d=JOBS/RESUME
        return not all((d/(f"e5_refit_e{ep}"+suffix)).is_file() for suffix in (".pt.verified","_test.npy.verified",".receipt.json"))
    nominal={};upper={}
    since=0 if rolled or fit.get("phase")=="training" else 6*60
    for ep in (1,2,3):
        if not pending_weights(ep):
            nominal[ep]=upper[ep]=now
            continue
        times=residual(fit,1,step,484,ep,7.60)
        epochs_left=max(1,ep-int(fit.get("epoch",1))+1)
        nominal[ep]=now+times[0]+since+epochs_left*65+5*60
        upper[ep]=now+times[1]+since+epochs_left*120+11*60
    er=residual(main,1,0,414,3,7.47);sr=residual(shadow,1,0,157,chosen,7.25)
    if read(JOBS/EVAL/"released.json"):
        eval_end=eval_upper=now
    else:
        remaining_epochs=max(0,3-int(main.get("epoch",1))+1) if main.get("phase")!="DONE" else 0
        extra=remaining_epochs*75+(0 if shadow else 150)
        replay_done=bool(list((JOBS/EVAL).glob("e5_trainonly_e*_test.npy.verified")))
        eval_end=now+er[0]+sr[0]+extra+(0 if replay_done else 8*60)
        eval_upper=now+er[1]+sr[1]+extra+(0 if replay_done else 15*60)
    point=book["estimated_cu"];bound=book["protected_cu"]
    rates=read(WORK/"incident_accounting.json")
    for s,typ,big in ((EVAL,eval_end,eval_upper),(RESUME,nominal[chosen],upper[chosen])):
        if read(JOBS/s/"released.json"):continue
        hw=book["sessions"].get(s,{}).get("gpu","A100")
        point+=max(0,typ-now)/3600*(rates["rate_high"] if hw=="A100-80GB" else rates["rate_standard"])
        bound+=max(0,big-now)/3600*book["protected_rates"][hw]
    return dict(utc=now,refit_status=fit,eval_status=main,shadow_status=shadow,
                selected_epoch=chosen if main.get("phase")=="DONE" else None,restore=rolled,
                fit_nominal_by_epoch=nominal,fit_upper_by_epoch=upper,alternative_upper_utc=eval_upper,
                finish_utc={RESUME:nominal[chosen],EVAL:eval_end},projected_point_cu=point,projected_upper_cu=bound,
                point_used=book["estimated_cu"],protected_used=book["protected_cu"],target_utc=TARGET,lock_utc=LOCK)

def supervise():
    from e5_rescue_policy import selected_test
    lookups={}
    while not LATCH.exists() and time.time()<LOCK+60:
        plan=estimate();atomic_json(WORK/"e5_rescue_status.json",plan)
        for s in (EVAL,RESUME):
            d=JOBS/s
            age=time.time()-read(d/"heartbeat.json").get("timestamp",time.time())
            if (d/"released.json").exists() or age<300 or time.time()-lookups.get(s,0)<120:continue
            lookups[s]=time.time()
            try:
                subprocess.run([str(Path(CLI).resolve().parent/"python3"),"scripts/restore_colab_mapping.py",s],
                    cwd=ROOT,timeout=35,capture_output=True,env=dict(os.environ,COLAB_CLI_REQUEST_TIMEOUT="12"))
            except subprocess.TimeoutExpired:pass
        selected=plan["selected_epoch"]
        fit=plan["refit_status"];d=JOBS/RESUME
        if selected and selected<3 and fit.get("phase")=="training" and fit.get("epoch",0)>selected and not (d/"released.json").exists():
            checks=[d/f"e5_refit_e{ep}.pt.verified" for ep in range(1,fit["epoch"])]
            try:pair=selected_test(Path("artifacts/jobs")/EVAL/f"e5_e{selected}_dev.npy",selected)
            except FileNotFoundError:pair=None
            if pair and pair["training"]==["train","dev"] and all(f.exists() for f in checks):
                atomic_json(d/"early_release_proof.json",dict(utc=time.time(),selected_epoch=selected,record=pair,checked=list(map(str,checks))))
                stop_one(RESUME,"selected epoch checkpoint and train+dev test scores verified; remaining epochs unused")
        if plan["projected_upper_cu"]>=STOP_AT and (d/"machine.json").is_file() and not (d/"released.json").exists():
            atomic_json(WORK/"e5_refit_unavailable.json",dict(utc=time.time(),reason="refit would use the protected E5 evaluation budget"))
            stop_one(RESUME,"protect evaluation, shadow and train-only alternative at the project CU guard")
        if all((JOBS/s/"released.json").is_file() for s in (EVAL,RESUME)):return
        time.sleep(25)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("allocate","dispatch","supervise"))
    args=p.parse_args()
    if args.action=="allocate":allocate()
    else:
        while not LATCH.exists() and time.time()<TARGET:
            try:
                if args.action=="supervise":supervise()
                else:dispatch()
                break
            except (OSError,RuntimeError,subprocess.SubprocessError) as exc:
                print("retry dispatch",str(exc)[-170:],flush=True);time.sleep(10)
