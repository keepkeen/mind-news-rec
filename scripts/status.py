"""One-shot local status; does not interact with Colab."""
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import json
import time
def clock(t):return datetime.fromtimestamp(t,ZoneInfo("Asia/Shanghai")).strftime("%m-%d %H:%M:%S")
art=Path("artifacts")
for p in sorted((art/"jobs").glob("mind*/heartbeat.json")):
    j=json.loads(p.read_text())
    closed=p.parent/"released.json"
    if closed.exists():
        print(p.parent.name,"STOPPED",clock(json.loads(closed.read_text())["stopped_utc"]))
    else:print(p.parent.name,"heartbeat",clock(j["timestamp"]),"ready",",".join(j["ready"]))
    for name,s in j["status"].items():
        phase=s["phase"]
        if phase=="training":detail=f"epoch {s['epoch']}/{s['epochs']} step {s['step']}/{s['steps']} epoch ETA "+clock(s["utc_epoch"]+s["epoch_eta_sec"])
        elif phase=="timing_smoke":detail=f"{s['done']}/{s['total']} measured {s['seconds_per_step']:.2f}s/step, {s['logical_positives']} positives"
        elif "done" in s and "total" in s:
            detail=f"{s['done']}/{s['total']}"+(" ETA "+clock(s["utc_epoch"]+s["eta_sec"]) if "eta_sec" in s else "")
        else:detail=str(s.get("error",s.get("AUC",s.get("child",""))))
        print(" ",name.removesuffix(".status.json"),phase,detail)
for p in [art/"measurement/progress.json",art/"local/coordinator.json",art/"fusion/today_progress.json",art/"fusion/fusion_three_train_prefix.progress.json"]:
    if p.exists(): print(p,json.loads(p.read_text()))
for kind in ("bge","upper","qwen","fullbge","qlora"):
    files=sorted((art/"jobs").glob(f"mind*/{kind}_e*_metrics.json"))
    if files:
        p=files[-1];j=json.loads(p.read_text())
        print("dev",p.name,{k:round(v,6) for k,v in j["all"].items() if k!="impressions"})
for p in sorted((art/"submissions").glob("*.validation.json")):
    j=json.loads(p.read_text());print(p,j["rows"],j["candidates"],j.get("dev",{}).get("all",{}),j["sha256"])

