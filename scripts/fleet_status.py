"""Read the six project heartbeats, official metrics and synchronized artifacts locally."""
import argparse
import json
import time
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from colab_jobs import ART,JOBS,NAMES,atomic_json
from project_budget import ledger
from fusion import branch,matched_refit

def read(p):
    return json.loads(p.read_text()) if p.is_file() else {}
def cst(t):return datetime.fromtimestamp(t,ZoneInfo("Asia/Shanghai")).strftime("%m-%d %H:%M")
def summary(save=False):
    queues=read(ART/"fleet/queues.json");budget=ledger()
    result={"as_of":cst(time.time()),"budget":{k:budget[k] for k in ("estimated_cu","protected_cu","limit_cu")},
            "sessions":{},"dev":{},"epoch_pairs":{}}
    for s in NAMES:
        d=JOBS/s;hb=read(d/"heartbeat.json");probe=read(d/"parallel_probe.json")
        active=read(d/"released.json");running={};states={v["job"]:v for v in hb.get("status",{}).values()}
        for k,v in probe.get("states",{}).items():
            if v.get("utc_epoch",0)>states.get(k,{}).get("utc_epoch",0):states[k]=v
        for k in queues.get(s,[]):
            q=states.get(k,{})
            key=q.get("child",k)
            st=states.get(key,q)
            if st.get("phase")=="MOVED":continue
            row={n:st[n] for n in ("phase","epoch","epochs","step","steps","loss","seconds_per_step","positive_pairs","peak_gib","error") if n in st}
            if st.get("phase")=="training":row["epoch_eta"]=cst(st["utc_epoch"]+st["epoch_eta_sec"])
            running[key]=row
        m=read(d/"machine.json")
        reconciled=budget["sessions"].get(s,{}).get("incident_reconciled",False)
        result["sessions"][s]=dict(device=m.get("gpu"),uuid=probe.get("uuid") or m.get("uuid"),last_heartbeat=cst(hb["timestamp"]) if hb else None,
            closed=bool(active) or reconciled,stopped_utc=active.get("stopped_utc"),incident_reconciled=reconciled,
            state={} if reconciled else running,ready=hb.get("ready",[]),progress=str(d/"heartbeat.json"))
    reports=read(ART/"fusion/dev_metrics.json")
    for k in ("bge","upper","qwen","blend"):
        if k in reports:result["dev"][k]=reports[k]
    for k in ("upper_s2","upper_s3","fullbge","fullbge_s2","qlora","asym4","e5"):
        for p in sorted(JOBS.glob(f"mind*/{k}_e*_metrics.json")):
            result["dev"][p.name[:-13]]=read(p)
    for k in ("bge","upper","qwen","upper_s2","upper_s3","fullbge","fullbge_s2","qlora","asym4","e5"):
        try:result["epoch_pairs"][k]=matched_refit(k,branch(k,"dev"))
        except FileNotFoundError:pass
    result["optional"]=read(ART/"fleet/asymmetric.json")
    result["today"]=read(ART/"fusion/today_progress.json")
    if save:atomic_json(ART/"fleet/status_at_handoff.json",result)
    print(result["as_of"],"CST. Project CU estimate %.2f / %.0f, protected %.2f"%(
        budget["estimated_cu"],budget["limit_cu"],budget["protected_cu"]))
    for s,x in result["sessions"].items():
        print(s,x["device"],"STOPPED" if x["closed"] else x["last_heartbeat"],json.dumps(x["state"],ensure_ascii=False))
    for k,v in result["dev"].items():
        a=v["all"]
        print("dev",k,*(round(a[m],6) for m in ("AUC","MRR","nDCG@5","nDCG@10")),"cold",v.get("cold_candidates_only",{}).get("AUC"),"short",round(v["history_le_5"]["AUC"],6))
    print("matched epochs",{k:v["epoch"] for k,v in result["epoch_pairs"].items()},"optional",result["optional"].get("stage"))
    return result
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--save",action="store_true");p.add_argument("--recovery",action="store_true")
    args=p.parse_args()
    if args.recovery:
        from recovery_status import snapshot
        result=snapshot()
        if args.save:atomic_json(ART/"round7/running_at_handoff.json",result)
        print(json.dumps(result,ensure_ascii=False,indent=2))
    else:summary(args.save)
