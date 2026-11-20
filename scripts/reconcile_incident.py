"""Freeze incident costs using only recorded device lifetimes, measured tariffs and historic account balances."""
import json
import re
from datetime import datetime
from pathlib import Path
from colab_jobs import ART,JOBS,NAMES,atomic_json
from project_budget import OLD,TZ,EST

R7=ART/"round7"
def historic_meter():
    log=Path.home()/".config/colab-cli/colab.log"
    stamp="";rows={}
    with log.open(errors="replace") as stream:
        for line in stream:
            tm=re.search(r"(2026-11-19 [0-9:]{8})",line[:65])
            if tm:stamp=tm.group(1)
            m=re.search(r'currentBalance["\x27]?\s*:\s*([\d.]+)',line)
            if stamp and m:
                rate=re.search(r'consumptionRateHourly["\x27]?\s*:\s*([\d.]+)',line)
                num=re.search(r'assignmentsCount["\x27]?\s*:\s*(\d+)',line)
                if rate and num:rows[stamp]=dict(time=datetime.fromisoformat(stamp).replace(tzinfo=TZ).timestamp(),
                    balance=float(m.group(1)),rate=float(rate.group(1)),assignments=int(num.group(1)))
    return list(rows.values())
def main():
    out=R7/"incident_accounting.json"
    if out.is_file():print(out.read_text());return
    meters=historic_meter()
    zero=next(x for x in meters if x["assignments"]==0 and x["time"]>datetime(2026,11,19,20,5,tzinfo=TZ).timestamp())
    six=next(x for x in meters if x["assignments"]==6 and abs(x["rate"]-46.83)<.03)
    high=json.loads((ART/"fleet/meter_rates.json").read_text())["observed_marginal"]
    standard=round((six["rate"]-3*high)/3,2)
    budgets=json.loads((R7/"budget_before_cleanup.json").read_text())
    rows={};before_last=before_est=before_upper=book_basis=book_typical=0.
    for s in NAMES[:8]:
        history=Path.home()/".config/colab-cli/history"/(s+".jsonl");start=end=last=None;reason=None
        with history.open() as stream:
            for l in stream:
                row=json.loads(l);t=datetime.fromisoformat(row["timestamp"]).timestamp();event=row["event_type"]
                if event=="session_created":start=t
                if event=="session_terminated":end=t;reason=row.get("reason")
                if event in ("file_operation","execution"):last=t
        job=JOBS/s;machine=json.loads((job/"machine.json").read_text());hw="A100-80GB" if "80GB" in machine["gpu"] else "A100"
        normal=s in ("mind_seed_eval_1119","mind_seed_fit_1119")
        if not start or not last:raise ValueError(s)
        upper=min(end or zero["time"],zero["time"]);lower=upper if normal else last
        estimate=upper if normal else min(upper,lower+2400)
        rate=high if hw=="A100-80GB" else standard
        costs=[(t-start)*rate/3600 for t in (lower,estimate,upper)]
        br=budgets["sessions"][s];lag=max(last,(job/"sync.json").stat().st_mtime,json.loads((job/"heartbeat.json").read_text())["timestamp"])
        bk=(max(0,(br["starting_utc"]+3600*br["device_hours"]) if normal else lag)-br["starting_utc"])*EST[hw]/3600
        book_basis+=bk;book_typical+=bk+(0 if normal else (estimate-lower)*EST[hw]/3600)
        rows[s]=dict(gpu=hw,start_utc=start,last_success_utc=last,first_known_absent_or_stopped_utc=upper,
                     exact_stop_receipt=normal,history_reason=reason,estimated_end_utc=estimate,
                     billed_rate=rate,cost_min=costs[0],cost_estimate=costs[1],cost_max=costs[2])
        before_last+=costs[0];before_est+=costs[1];before_upper+=costs[2]
    # Before loss, every then-active project device was still reachable. The entire joint account debit
    # thereafter is an upper bound on additional project usage, not an attribution to this project.
    cutoff=min(v["last_success_utc"] for v in rows.values() if not v["exact_stop_receipt"])
    bounds=[]
    for m in meters:
        if m["time"]>=cutoff:continue
        earned=sum(max(0,min(m["time"],v["first_known_absent_or_stopped_utc"])-v["start_utc"])*v["billed_rate"]/3600 for v in rows.values())
        bounds.append((earned+m["balance"]-zero["balance"],m,earned))
    joint,anchor,earned=min(bounds,key=lambda r:r[0])
    result=dict(rows=rows,confirmed_device_time_cu=before_last,estimated_actual_cu=before_est,
        estimate_assumption="40 minutes after each last confirmed interaction, capped by first known absence",
        direct_duration_upper_cu=before_upper,account_constrained_upper_cu=min(joint,before_upper),
        account_bound_assumption="no top-up or expiry between these two same-evening balance readings",
        account_anchor=anchor,attributable_at_anchor=earned,empty_account_receipt=zero,
        project_budget_rate_through_success=book_basis,project_budget_rate_idle_estimate=book_typical,
        raw_before_cleanup=budgets["estimated_cu"],raw_protected_before_cleanup=budgets["protected_cu"],
        rate_standard=standard,rate_high=high,exact_project_invoice_available=False)
    atomic_json(out,result)
    print({k:v for k,v in result.items() if isinstance(v,(float,int))})
if __name__=="__main__":main()
