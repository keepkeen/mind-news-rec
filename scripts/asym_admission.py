"""Admit a one-epoch optional matched pair only with measured speed, time and project CU headroom."""
from datetime import datetime
from zoneinfo import ZoneInfo
import concurrent.futures as cf
import json
import time
from colab_jobs import ART,JOBS,atomic_json,rex,response
from project_budget import ledger,PROTECTED,STOP_AT

PAIR={"mind_seed_eval_1119":"asym4","mind_seed_fit_1119":"asym4_refit"}
ZONE=ZoneInfo("Asia/Shanghai")
def when(day,hour,minute):
    return datetime(2026,11,day,hour,minute,tzinfo=ZONE).timestamp()
# Dedicated runs include full evaluation, their post-selected early shadows and remaining synchronization.
CORE_END={"mind_h100":when(19,19,30),"mind_plm_1119":when(20,0,30),"mind_fullfit_1119":when(19,19,45),
          "mind_qfit_1119":when(19,23,10),"mind_seed_eval_1119":when(19,16,10),"mind_seed_fit_1119":when(19,16,0)}
LOCK=when(20,4,30)
def getpilot(session,kind):
    d=JOBS/session
    f=d/(kind+"_pilot.json")
    if f.exists():return json.loads(f.read_text())
    hb=d/"heartbeat.json"
    if not hb.exists():return
    st=json.loads(hb.read_text())["status"].get(kind+".status.json",{})
    if st.get("phase")=="awaiting_budget_decision":
        out=response(rex(session,f"import json\nfrom pathlib import Path\nprint('@@MIND '+json.dumps(json.loads(Path('/content/mind/export/{kind}_pilot.json').read_text())))",55,2))
        atomic_json(f,out);return out
def deliver(session,decision):
    if (JOBS/session/"released.json").exists():return
    cmd=f"""from pathlib import Path
import json
f=Path('/content/mind/run/asym_admission.json');tmp=f.with_suffix('.tmp')
tmp.write_text(json.dumps({decision!r}));tmp.replace(f)
print('@@MIND '+json.dumps({{'run':{decision['run']!r}}}))
"""
    response(rex(session,cmd,70,3))
def main():
    dst=ART/"fleet/asymmetric.json"
    if dst.exists():
        prior=json.loads(dst.read_text())
        if prior.get("stage") in ("ADMITTED","SKIPPED"):
            with cf.ThreadPoolExecutor() as pool:list(pool.map(lambda s:deliver(s,prior),PAIR))
            return
    while True:
        pilots={s:v for s,k in PAIR.items() if (v:=getpilot(s,k))}
        now=time.time();budget=ledger(now)
        if pilots or now>LOCK-3600 or (ART/"fleet/no_allocation").exists():break
        if all((JOBS/s/"released.json").exists() for s in PAIR):
            atomic_json(dst,dict(stage="CLOSED",reason="seed runtimes completed before an eligible pilot",time=now));return
        atomic_json(dst,dict(stage="QUEUED_AFTER_NEW_SEEDS",machines=PAIR,gate_hours_per_epoch=6,
                 cached_history_sha256="f7588663b9d9f4fd44cc740739b48fd43bc320988e9e1b505ac7f4a5fd1739e2"))
        time.sleep(60)
    rates={s:PROTECTED[v["gpu"]] for s,v in budget["sessions"].items()}
    reserve=sum(max(0,until-now)/3600*rates[s] for s,until in CORE_END.items() if budget["sessions"][s]["state"]=="active")
    worst_step=max((p["steady_step_seconds"] for p in pilots.values()),default=1e6)
    def hours(steps):return (worst_step*steps*1.10+180)/3600
    ev,fit,shadow=hours(104),hours(121),hours(40)
    extras=(ev+shadow+.55)*rates["mind_seed_eval_1119"]+(fit+.45)*rates["mind_seed_fit_1119"]
    protected=budget["protected_cu"]+reserve+extras
    run=bool(pilots) and ev<=6 and fit<=6 and protected<STOP_AT and now+(max(ev+shadow,fit)+.8)*3600<LOCK and not (ART/"fleet/no_allocation").exists()
    decision=dict(stage="ADMITTED" if run else "SKIPPED",run=run,reason="measured epoch, deadline and project CU gates",
        measured_sessions=list(pilots),step_seconds=worst_step,evaluation_hours=ev,refit_hours=fit,shadow_hours=shadow,
        incurred_protected_cu=budget["protected_cu"],core_remaining_reserve_cu=reserve,optional_reserve_cu=extras,
        projected_protected_cu=protected,protected_limit=STOP_AT,decided_utc=now)
    atomic_json(dst,decision)
    with cf.ThreadPoolExecutor() as pool:list(pool.map(lambda s:deliver(s,decision),PAIR))
if __name__=="__main__":
    while True:
        try:main();break
        except (OSError,ValueError,RuntimeError) as e:
            print(str(e)[-200:],flush=True);time.sleep(35)
