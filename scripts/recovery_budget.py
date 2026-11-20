"""Budget and deadline admission for the three new sessions only."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
import json
import shutil
import sys
import time
from zoneinfo import ZoneInfo
from colab_alerts import LATCH
from colab_jobs import ART,JOBS,atomic_json
from e5_admission import deliver,pilot,read
from project_budget import STOP_AT,ledger,protected_rates,stop_one
from recovery_watchdog import SESSIONS,current_roles

R=ART/"round7";TARGET=datetime(2026,11,20,4,15,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
LOCK=TARGET+900
def forecast(a,b,now=None):
    now=time.time() if now is None else now;book=ledger(now);rates=protected_rates()
    hist=read(R/"incident_accounting.json");old=read(ART/"round6/admission.json")
    def costs(v,steps,early=False):
        times=v.get("shadow_step_seconds" if early else "step_seconds",[])
        sec=max(times[1:]) if len(times)>1 else (6.95 if early else 7.2)
        # A 10% step allowance plus explicit full-state transfer / evaluation / startup allowance.
        return 3*steps*sec*1.10/3600
    ev=costs(a,414)+costs(a,157,True)+.48;fit=costs(b,484)+.43
    bg="mind_r7_bge";bs=read(JOBS/bg/"heartbeat.json").get("status",{})
    shadow=bs.get("fullbge_s2_shadow.status.json",{})
    if (JOBS/bg/"released.json").exists():bh=0.
    elif shadow.get("phase")=="DONE":bh=.13
    elif shadow.get("phase")=="training":
        left=(2-shadow["epoch"])*157+157-shadow["step"]
        bh=left*max(7.4,shadow["sec_per_step"])*1.10/3600+.18
    else:bh=2*157*7.40*1.10/3600+.24
    full_hours=dict(zip(SESSIONS,(ev,fit,bh)));hours=dict(full_hours)
    sent=read(R/"admission.json").get("delivered_roles",{})
    for s in SESSIONS[:2]:
        ack=sent.get(s,{})
        hb=read(JOBS/s/"heartbeat.json")
        if ack.get("run") and now-hb.get("timestamp",0)<300:
            hours[s]=max(.08,hours[s]-max(0,now-ack["utc"])/3600)
    point=book["estimated_cu"];safe=book["protected_cu"]
    for s,h in hours.items():
        shape=book["sessions"].get(s,{}).get("gpu","A100" if s==bg else "A100-80GB")
        if not (JOBS/s/"machine.json").is_file():h+=.18
        point+=h*(hist["rate_high"] if shape=="A100-80GB" else hist["rate_standard"]);safe+=h*rates[shape]
    return dict(utc=now,hours=hours,full_hours=full_hours,finish_utc={s:now+h*3600 for s,h in hours.items()},target=TARGET,
        measured_or_prior=dict(evaluation=a.get("step_seconds"),refit=b.get("step_seconds"),shadow=a.get("shadow_step_seconds")),
        spent_point=book["estimated_cu"],spent_upper=book["protected_cu"],projected_total=point,projected_upper=safe,
        E5_only_upper=safe-bh*rates[book["sessions"].get(bg,{}).get("gpu","A100")],guard=STOP_AT,cap=450)

def main():
    dest=R/"admission.json";slots=SESSIONS[:2]
    past=(read(JOBS/"mind_e5_eval_1119/e5_pilot.json"),read(JOBS/"mind_e5_fit_1119/e5_refit_pilot.json"))
    prior=read(dest);sent=prior.get("delivered_roles",{})
    if prior.get("delivered") and not sent:return
    while not LATCH.exists() and time.time()<TARGET:
        probes=[pilot(slots[0],"e5"),pilot(slots[1],"e5_refit")]
        if not any(probes):time.sleep(8);continue
        d=forecast(probes[0] or past[0],probes[1] or past[1])
        in_time=max(d["finish_utc"][s] for s in slots)<TARGET
        d.update(run=in_time and d["E5_only_upper"]<STOP_AT,keep_bge=d["projected_upper"]<STOP_AT,
                 current_pilot={s:bool(v) for s,v in zip(slots,probes)},
                 reason="each E5 role starts after its own pilot; the other role uses its R6 measurement until its new pilot arrives")
        if d["run"] and not d["keep_bge"]:
            (R/"stop_bge").touch()
            if (JOBS/SESSIONS[2]/"machine.json").exists() and not (JOBS/SESSIONS[2]/"released.json").exists():
                stop_one(SESSIONS[2],"preserve recovery budget for E5")
        for role,probe in zip(slots,probes):
            if probe and role not in sent:
                deliver(role,d);sent[role]=dict(run=d["run"],utc=time.time(),projected_upper=d["projected_upper"])
        d.update(delivered_roles=sent,delivered=len(sent)==2)
        atomic_json(dest,d)
        if len(sent)==2:
            print("two decisions delivered",d["projected_upper"],flush=True);return
        time.sleep(8)

def guard():
    while not LATCH.exists():
        b=ledger();atomic_json(ART/"fleet/budget.json",b)
        from recovery_sync import OUTPUT_RESERVE,ROLL
        roles=current_roles()
        for s in roles:
            d=JOBS/s;receipt=read(d/"released.json");phase=receipt.get("queue_phase",{})
            if not phase or any(v!="DONE" for v in phase.values()) or time.time()-receipt["stopped_utc"]<30:continue
            for name in receipt.get("terminal_optional_rolling",[]):
                if ROLL.fullmatch(name):
                    for suffix in (".receiving",".transfer.json"):(d/("."+name+suffix)).unlink(missing_ok=True)
            if shutil.disk_usage(d).free<OUTPUT_RESERVE:
                latest=read(d/"rolling_latest.json");name=latest.get("name","")
                if ROLL.fullmatch(name) and (d/name).exists() and not (d/"terminal_rolling_reclaimed.json").exists():
                    atomic_json(d/"terminal_rolling_reclaimed.json",dict(**latest,reason="all epochs and required scores verified before release",utc=time.time()))
                    (d/name).unlink();(d/(name+".verified")).unlink(missing_ok=True)
        active=[s for s in roles if b["sessions"].get(s,{}).get("state")=="active"]
        admit=read(R/"admission.json")
        if admit.get("run") and SESSIONS[2] in active and not (R/"stop_bge").is_file():
            pilots=[read(JOBS/s/(job+"_pilot.json")) for s,job in zip(SESSIONS[:2],("e5","e5_refit"))]
            past=[read(JOBS/s/(job+"_pilot.json")) for s,job in zip(("mind_e5_eval_1119","mind_e5_fit_1119"),("e5","e5_refit"))]
            f=forecast(pilots[0] or past[0],pilots[1] or past[1])
            if f["E5_only_upper"]+2>=STOP_AT:
                (R/"stop_bge").touch();stop_one(SESSIONS[2],"yield recovery budget to E5")
        if b["protected_cu"]>=STOP_AT or (time.time()>=LOCK and active):
            (R/"no_new_assignments").touch()
            with ThreadPoolExecutor() as p:list(p.map(lambda s:stop_one(s,"recovery CU guard or 04:30 lock"),active))
        if all((JOBS/s/"released.json").exists() for s in roles):return
        time.sleep(20)
if __name__=="__main__":
    while not LATCH.exists():
        try:
            if "--guard" in sys.argv:guard()
            else:main()
            break
        except (OSError,ValueError,RuntimeError) as e:
            print(repr(e)[-190:],flush=True);time.sleep(8)
