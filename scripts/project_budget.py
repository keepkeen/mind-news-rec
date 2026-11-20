"""Project-only device-hour ledger and independent stop guard. No other session is eligible."""
from datetime import datetime
from zoneinfo import ZoneInfo
import concurrent.futures as cf
import json
import os
import sys
import time
from colab_jobs import ART,JOBS,NAMES,atomic_json,call

LIMIT=450.
STOP_AT=432.
# A100 40GB standard is approximately 6.7 CU/h in this account. Higher planning bands allow billing/stop latency.
EST={"A100":6.72,"A100-80GB":10.,"H100":12.};PROTECTED={"A100":8.,"A100-80GB":12.,"H100":18.}
TZ=ZoneInfo("Asia/Shanghai")
OLD={"mind_h100":"2026-11-19 13:09","mind_plm_1119":"2026-11-19 13:15"}
def info(p):
    return json.loads(p.read_text()) if p.exists() else {}
def protected_rates():
    result=PROTECTED.copy()
    calibration=info(ART/"fleet/meter_rates.json")
    for shape,amount in calibration.get("protected_rates",{}).items():
        if shape in EST and amount>=EST[shape]:result[shape]=amount
    return result
def ledger(now=None):
    now=time.time() if now is None else now
    rows={};rates=protected_rates()
    incident=info(ART/"round7/incident_accounting.json")
    reconcile=bool(incident) and now>=incident["empty_account_receipt"]["time"]
    for s in NAMES:
        d=JOBS/s;m=info(d/"machine.json");a=info(d/"allocation.json");end=info(d/"released.json")
        if reconcile and s in incident["rows"]:
            v=incident["rows"][s]
            rows[s]=dict(state="released",gpu=v["gpu"],starting_utc=v["start_utc"],
                device_hours=(v["estimated_end_utc"]-v["start_utc"])/3600,estimated_cu=v["cost_estimate"],
                protected_cu=v["cost_estimate"],incident_reconciled=True,exact_stop=v["exact_stop_receipt"])
            continue
        if s in OLD:start=datetime.strptime(OLD[s],"%Y-%m-%d %H:%M").replace(tzinfo=TZ).timestamp()
        elif a:start=a.get("first_reconciled_request",a["first_request"])
        else:continue
        if s.startswith("mind_r7_"):
            absence=info(d/"billing_presence.json")
            if absence.get("notebook")==a.get("notebook") and absence.get("absent_at",0)<=m.get("allocated_utc",now):
                start=max(start,absence["absent_at"])
        device=m.get("gpu","A100-80GB" if a.get("high_ram") and a.get("gpu")=="A100" else a.get("gpu","A100"))
        hw="H100" if "H100" in device else "A100-80GB" if ("80GB" in device or "81920" in device) else "A100"
        state="released" if end and end.get("stopped_utc",now)<=now else ("active" if m else "allocation_pending")
        # Pending assignments are conservatively reserved from the first attempt. Confirmed refusals with no VM incur no charge.
        until=min(now,end.get("stopped_utc",now))
        boot=info(d/"boot.json")
        hours=max(0.,until-start)/3600
        if state=="allocation_pending" and boot.get("stage")=="CAPACITY_UNAVAILABLE":
            state="unallocated";hours=0.
        effective_start=m.get("allocated_utc",start) if s.startswith("mind_r7_") else start
        effective_rate=(incident["rate_high"] if hw=="A100-80GB" else incident["rate_standard"]) if reconcile and hw in ("A100","A100-80GB") else EST[hw]
        rows[s]=dict(state=state,gpu=hw,starting_utc=start,device_hours=hours,
                     estimated_cu=max(0,until-effective_start)/3600*effective_rate if hours and (m or not s.startswith("mind_r7_")) else 0.,
                     protected_cu=hours*rates[hw])
    historical_margin=(incident["account_constrained_upper_cu"]-incident["estimated_actual_cu"]) if reconcile else 0.
    return dict(utc=now,limit_cu=LIMIT,emergency_stop_protected_cu=STOP_AT,account_others_excluded=True,
                estimated_cu=sum(x["estimated_cu"] for x in rows.values()),
                protected_cu=sum(x["protected_cu"] for x in rows.values())+historical_margin,
                historic_spend_uncertainty_reserve=historical_margin,
                original_rate_upper_cu=sum(x["device_hours"]*PROTECTED[x["gpu"]] for x in rows.values()),
                protected_rates=rates,sessions=rows)
def stop_one(s,reason="project CU guard"):
    if s not in NAMES:raise ValueError(s)
    d=JOBS/s
    if (d/"released.json").exists():return
    try:out=call(["stop","-s",s],65,2)
    except RuntimeError as e:
        if "not found" not in str(e).lower():raise
        # An in-flight request may still acquire its stable notebook ID. Keep checking until boot exits.
        if not (d/"machine.json").exists():return
        out="already absent"
    atomic_json(d/"released.json",dict(session=s,stopped_utc=time.time(),reason=reason,reply=out[-180:]))
    (d/"stop_monitor").touch()
def watch():
    root=ART/"fleet";root.mkdir(parents=True,exist_ok=True)
    while True:
        report=ledger();atomic_json(root/"budget.json",report)
        if report["protected_cu"]>=STOP_AT or (root/"no_allocation").exists():
            (root/"no_allocation").touch()
            names=[s for s,v in report["sessions"].items() if v["state"] in ("active","allocation_pending")]
            with cf.ThreadPoolExecutor(max_workers=6) as pool:
                for future in [pool.submit(stop_one,s) for s in names]:
                    try:future.result()
                    except Exception as exc:print("scoped stop retry",str(exc)[-160:],flush=True)
        if len(report["sessions"])==len(NAMES) and all(x["state"] in ("released","unallocated") for x in report["sessions"].values()):
            return
        time.sleep(45)
if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="watch":watch()
    else:print(json.dumps(ledger(),indent=2))

