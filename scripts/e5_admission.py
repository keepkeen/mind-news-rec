"""Reserve the existing 1163/0.6B work, then admit both E5 jobs using their own timed pilots."""
import concurrent.futures as cf
from datetime import datetime
import json
import math
import sys
import time
from zoneinfo import ZoneInfo
from colab_jobs import ART,JOBS,NAMES,atomic_json,rex,response
from project_budget import EST,STOP_AT,ledger,protected_rates,stop_one

R6=ART/"round6";LOCK=datetime(2026,11,20,4,30,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
R6_E5_SESSIONS={"mind_e5_eval_1119","mind_e5_fit_1119"}
def read(p):
    return json.loads(p.read_text()) if p.is_file() else {}

def measure_rates():
    before=read(R6/"meter_before.json");after=read(R6/"meter_after.json")
    if not before or not after or before["other_fingerprint"]!=after["other_fingerprint"]:return
    old,new=before["own"],after["own"]
    added=set(new)-set(old)
    if not added or not added<=R6_E5_SESSIONS or any(new.get(s)!=v for s,v in old.items()):return
    shapes={tuple(new[s]) for s in added}
    if len(shapes)!=1 or after["active"]-before["active"]!=len(added):return
    acc,shape=shapes.pop()
    device="H100" if acc=="H100" else "A100-80GB" if acc=="A100" and shape=="HIGH_RAM" else "A100" if acc=="A100" else ""
    if not device:return
    marginal=(after["account_cu_per_hour"]-before["account_cu_per_hour"])/len(added)
    # Cross-check the clean marginal quote against the earlier estimate before replacing an unsourced planning band.
    if not .8*EST[device]<=marginal<=1.2*EST[device]:return
    # Keep the higher of the previous shape estimate and the observed tariff plus 12% and meter rounding.
    upper=math.ceil(100*max(EST[device],1.12*(marginal+.02)))/100
    atomic_json(ART/"fleet/meter_rates.json",dict(protected_rates={device:upper},clean_same_inventory=True,
        prior_shape_estimate=EST[device],observed_marginal=marginal,source_before=str(R6/"meter_before.json"),
        source_after=str(R6/"meter_after.json"),added=sorted(added),same_day_rate_premium=.12))

def profile(s,name,default):
    return read(JOBS/s/(name+"_smoke.json")).get("steady_step_seconds",default)
def core_forecast():
    estimates={}
    def fit(s,name,epochs,steps,base,per_epoch,start):
        stages=read(JOBS/s/"heartbeat.json").get("status",{})
        st=stages.get(name+".status.json",{})
        phase=st.get("phase")
        if phase in ("DONE","FAILED","SKIPPED","MOVED"):return 0.
        ep=st.get("epoch",1)
        step=st.get("step",0) if phase=="training" else steps if phase=="refresh_for_evaluation" else 0
        sec=max(base,st.get("sec_per_step",0)) if step>=4 else base
        iterations=max(0,(epochs-ep+1)*steps-step)
        return (iterations*sec*1.10+max(1,epochs-ep+1)*per_epoch+(0 if phase in ("training","refresh_for_evaluation") else start))/3600
    for s,kind,ep,steps,stage_cost in [
            ("mind_h100","fullbge_s2",3,414,65),("mind_plm_1119","qlora",2,104,170),
            ("mind_fullfit_1119","fullbge_s2_refit",3,484,90),("mind_qfit_1119","qlora_refit",2,121,160)]:
        if (JOBS/s/"released.json").is_file():estimates[s]=0.;continue
        base=profile(s,kind,98.2 if "qlora" in kind else 7.7)
        own=fit(s,kind,ep,steps,base,stage_cost,300)
        st=read(JOBS/s/"heartbeat.json").get("status",{})
        if kind in ("fullbge_s2","qlora"):
            selected=read(JOBS/s/(kind+"_best.json")).get("best_epoch",ep) if st.get(kind+".status.json",{}).get("phase")=="DONE" else ep
            shadow=kind+"_shadow";shadow_steps=40 if kind=="qlora" else 157
            shadow_sec=profile(s,shadow,base if kind=="qlora" else profile(s,"fullbge_shadow",base))
            own+=fit(s,shadow,selected,shadow_steps,shadow_sec,45,420 if kind=="qlora" else 140)
        elif kind=="qlora_refit" and st.get("qlora_checkpoint_replay.status.json",{}).get("phase")!="DONE":
            own+=.15
        estimates[s]=own+.13
    return estimates

def pilot(s,job):
    p=JOBS/s/(job+"_pilot.json")
    if p.is_file():return read(p)
    hb=read(JOBS/s/"heartbeat.json").get("status",{})
    if hb.get(job+".status.json",{}).get("phase")=="awaiting_paired_admission":
        data=response(rex(s,f"import json,pathlib\np=pathlib.Path('/content/mind/export/{job}_pilot.json')\nprint('@@MIND '+json.dumps(json.loads(p.read_text())))",55,2))
        atomic_json(p,data);return data
    return {}

def forecast(placement,probes):
    now=time.time();book=ledger(now);rates=protected_rates();core=core_forecast()
    a,b=probes["evaluation"],probes["refit"]
    early=max(a["shadow_step_seconds"][1:])
    eh=(3*a["steps_per_epoch"]*max(a["step_seconds"][1:])*1.10+3*a["shadow_steps"]*early*1.10)/3600+.36
    fh=3*b["steps_per_epoch"]*max(b["step_seconds"][1:])*1.10/3600+.30
    hours={"evaluation":eh,"refit":fh}
    nominal=book["estimated_cu"];safe=book["protected_cu"]
    for s,h in core.items():
        hw=book["sessions"][s]["gpu"];nominal+=h*EST[hw];safe+=h*rates[hw]
    finished={}
    for role,s in placement.items():
        h=hours[role];hw=book["sessions"][s]["gpu"];nominal+=h*EST[hw];safe+=h*rates[hw]
        finished[role]=now+(h+core.get(s,0))*3600
    deadline_ok=max(finished.values())+30*60<LOCK
    return dict(time=now,expected_device_hours=hours,upper_finish_utc=finished,core_reserve_hours=core,
            incurred_estimated_cu=book["estimated_cu"],incurred_protected_cu=book["protected_cu"],
            original_rate_upper_cu=book["original_rate_upper_cu"],projected_total_cu=nominal,
            projected_protected_cu=safe,protected_rates=rates,guard_cu=STOP_AT,limit_cu=450,
            run=safe<STOP_AT and deadline_ok and not (ART/"fleet/no_allocation").exists(),deadline_ok=deadline_ok,
            steps={k:v["steps_per_epoch"] for k,v in probes.items()},step_seconds={k:v["step_seconds"] for k,v in probes.items()},
            shadow_steps=a["shadow_steps"],shadow_step_seconds=a["shadow_step_seconds"],
            peak_gib={k:v["peak_gib"] for k,v in probes.items()},placement=placement)
def deliver(s,decision):
    code=f"""import pathlib,json
p=pathlib.Path('/content/mind/run/e5_admission.json');tmp=p.with_suffix('.tmp')
tmp.write_text(json.dumps({decision!r}));tmp.replace(p)
print('@@MIND '+json.dumps({{'run':{decision['run']!r}}}))
"""
    response(rex(s,code,60,3))
def run():
    output=R6/"admission.json"
    if read(output).get("delivered"):return
    while True:
        placement=read(R6/"placement.json");probes={}
        for mode,s in placement.items():probes[mode]=pilot(s,"e5_refit" if mode=="refit" else "e5")
        if len(probes)==2 and all(probes.values()):break
        if time.time()>LOCK:return
        time.sleep(12)
    decision=read(output)
    if not decision:
        measure_rates()
        decision=forecast(placement,probes)
        decision["reason"]="paired measured cost and wall time" if decision["run"] else "project reserve or 04:30 deadline"
        atomic_json(output,decision)
    with cf.ThreadPoolExecutor() as pool:list(pool.map(lambda s:deliver(s,decision),placement.values()))
    decision["delivered"]=True;atomic_json(output,decision)
    print("ADMITTED" if decision["run"] else "SKIPPED",decision["projected_protected_cu"],flush=True)

def reserve_watch():
    path=R6/"priority_guard.json"
    while True:
        plan=read(R6/"admission.json");slots=read(R6/"placement.json")
        if plan and not plan["run"]:return
        if slots and all((JOBS/s/"released.json").exists() for s in slots.values()):return
        if plan.get("delivered"):
            book=ledger();rates=protected_rates();core=core_forecast()
            required=sum(h*rates[book["sessions"][s]["gpu"]] for s,h in core.items())
            must_stop=book["protected_cu"]+required+2.0>=STOP_AT or time.time()>=LOCK
            atomic_json(path,dict(time=time.time(),protected_used=book["protected_cu"],protected_remaining_for_priority=required,
                         stop_optional=must_stop,policy="preserve full BGE 1163 and Qwen, plus a two-CU stop lead"))
            if must_stop:
                for s in slots.values():
                    if s not in R6_E5_SESSIONS:raise RuntimeError("Only dedicated E5 machines may be shed")
                    stop_one(s,"optional E5 compute released to preserve the priority reserve or 04:30 lock")
                return
        time.sleep(45)
if __name__=="__main__":
    while True:
        try:
            if "--reserve-watch" in sys.argv:reserve_watch()
            else:run()
            break
        except (OSError,ValueError,RuntimeError) as e:print("retry",repr(e)[-220:],flush=True);time.sleep(15)
