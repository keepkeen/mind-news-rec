"""Read the recovery devices, their local checkpoint receipts, deadline and project bill."""
import json
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from colab_jobs import ART,JOBS
from project_budget import ledger
from recovery_watchdog import current_roles

def read(path):
    return json.loads(path.read_text()) if path.is_file() else {}

def local_time(t):
    return datetime.fromtimestamp(t,ZoneInfo("Asia/Shanghai")).strftime("%m-%d %H:%M:%S") if t else None

def snapshot():
    now=time.time()
    work=ART/"round7"
    plan=read(work/"admission.json")
    rescue=read(work/"e5_rescue_status.json")
    eta=dict(plan.get("finish_utc",{}));eta.update(rescue.get("finish_utc",{}))
    book=ledger(now)
    alerts=read(work/"alerts_status.json")
    devices={}
    for session,queue in zip(current_roles(),("e5_evaluation_queue","e5_refit_queue","bge_recovery_queue"),strict=True):
        d=JOBS/session
        hb=read(d/"heartbeat.json")
        states={s["job"]:s for s in hb.get("status",{}).values()}
        for key,s in read(d/"parallel_probe.json").get("states",{}).items():
            if s.get("utc_epoch",0)>states.get(key,{}).get("utc_epoch",0):states[key]=s
        q=states.get(queue,{})
        s=states.get(q.get("child"),q) if q.get("phase")=="running" else q
        if session.endswith("e5_eval") and q.get("phase")=="DONE" and states.get("e5_trainonly_replay"):
            s=states["e5_trainonly_replay"]
        m=read(d/"machine.json")
        devices[session]=dict(device=m.get("gpu"),uuid=m.get("uuid"),released=bool(read(d/"released.json")),
            state={k:s[k] for k in ("job","phase","epoch","epochs","step","steps","loss","sec_per_step","utc_epoch") if k in s},
            heartbeat_age_seconds=round(now-hb["timestamp"]) if hb else None,
            rolling_checkpoint=read(d/"rolling_latest.json").get("name"),
            epoch_checkpoints=[p.name.removesuffix(".verified") for p in sorted(d.glob("*_e[0-9].pt.verified"))],
            estimated_local_delivery_cst=local_time(eta.get(session)),progress=str(d/"heartbeat.json"))
    return dict(local_time=local_time(now),utc=now,sessions=devices,
        budget=dict(estimated_used=book["estimated_cu"],protected_used=book["protected_cu"],
                    projected_at_admission=rescue.get("projected_point_cu",plan.get("projected_total")),
                    projected_conservative=rescue.get("projected_upper_cu",plan.get("projected_upper")),
                    cap=book["limit_cu"]),auth_latched=alerts.get("auth_latched"),
        lock=read(ART/"local/coordinator.json").get("stage"),decision=read(work/"final_decision.json"),
        fallback=read(work/"independent_validation.json").get("sha256"))
