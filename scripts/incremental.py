"""Continuously publish dev-qualified, validated MIND submissions and notify the local operator."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from colab_jobs import ROOT,atomic_json
from coordinate import LOCK,SAFE_END
from fusion import branch,shadow_path,matched_refit,refit_tree,score_test
from e5_rescue_policy import content_training,scopes
from incremental_policy import (EXTRAS,MANIFEST,MIN_DEV,R5,STATED_BASELINE,SUB,assess,catalog,completed,read,
                                refit_records,sha256,signature,validated_copy)

STATUS=R5/"incremental_status.json"
BUILDS=R5/"builds"

def empty_manifest():
    return dict(reference_dev=STATED_BASELINE,minimum_full_dev=MIN_DEV,selection="full user half A; confirm on B",
                packages={},preferred={})

def manifest():
    return read(MANIFEST) if MANIFEST.is_file() else empty_manifest()

def stem(spec):
    return "fusion_"+spec["gate"]["family"]+"_"+spec["key"][:16]+"_prefix"

def build(specfile):
    folder=specfile.parent
    with (folder/"build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        spec=read(specfile);trial=spec["trial"];g=spec["gate"];name=stem(spec)
        if not g["release"] or trial["full_dev"]<MIN_DEV:raise ValueError("Unqualified package")
        records=refit_records(trial)
        if signature(trial,records)!=spec["key"]:raise ValueError("Changed inputs after selection")
        staged=folder/(name+".zip")
        receipt=staged.with_suffix(".validation.json")
        if receipt.is_file() and staged.is_file():
            info=read(receipt)
            if info.get("incremental_key")==spec["key"] and sha256(staged)==info["sha256"]:return
        pol=read(Path(trial["dest"])/"candidate_policy.json")
        atomic_json(folder/"policy.json",pol)
        atomic_json(folder/"worker.json",dict(stage="refitting_tree_on_Nov_12_to_15",started=time.time()))
        refit_tree(folder)
        score_test(folder,"policy.json","ranker.lgb",name,refit_base=True,stage="scoring_incremental_test",output_dir=folder,pinned_records=records)
        checked=read(receipt)
        if checked["matched_refit"]!=records or checked["content_training"]!=content_training(records) or checked["future_exposure"]:
            raise ValueError("Submission content provenance changed during scoring")
        checked.update(incremental_key=spec["key"],ranker_sha256=sha256(folder/"ranker.lgb") if pol["weights"][-1] else None,
                       metrics={k:trial[k] for k in ("full_dev","selection_A","heldout_B")},gate=g,models=trial["models"])
        atomic_json(receipt,checked)
        (folder/(name+"_test.npy")).unlink(missing_ok=True)
        atomic_json(folder/"worker.json",dict(stage="VALIDATED",sha256=checked["sha256"],zip=str(staged),time=time.time()))

def notify(entry,alias):
    body=str(Path(entry["zip"]).relative_to(ROOT))+"\nSHA-256 "+entry["sha256"]
    subtitle=f"dev {entry['full_dev']:.6f}  B {entry['heldout_B']:.6f}"+("  "+alias.name if alias else "")
    script='on run argv\n display notification (item 1 of argv) with title "MIND 提交包已校验" subtitle (item 2 of argv)\nend run'
    try:
        result=subprocess.run(["/usr/bin/osascript","-e",script,body,subtitle],capture_output=True,timeout=8)
        return result.returncode==0
    except (OSError,subprocess.TimeoutExpired):return False

def register(spec,staged,ledger):
    key=spec["key"];trial=spec["trial"];g=spec["gate"];family=g["family"]
    if key in ledger["packages"]:return
    receipt=read(staged.with_suffix(".validation.json"))
    expected_metrics={k:trial[k] for k in ("full_dev","selection_A","heldout_B")}
    if (not g["release"] or trial["full_dev"]<MIN_DEV or receipt.get("incremental_key")!=key
            or receipt.get("matched_refit")!=spec["refs"] or receipt.get("content_training")!=content_training(spec["refs"])
            or receipt.get("metrics")!=expected_metrics or receipt.get("future_exposure") is not False
            or receipt.get("tree_fit_days")!="Nov 12–15"):
        raise ValueError("Mismatched qualification or validation receipt")
    dest=SUB/"incremental"/(stem(spec)+".zip")
    if dest.exists() and sha256(dest)!=receipt["sha256"]:raise ValueError(f"Immutable archive collision: {dest}")
    actual=validated_copy(staged,dest,"complete dev gain and held-out user criterion passed")
    entry=dict(key=key,family=family,models=trial["models"],weights=trial["weights"],full_dev=trial["full_dev"],
               selection_A=trial["selection_A"],heldout_B=trial["heldout_B"],gate=g,zip=str(dest),sha256=actual["sha256"],
               validation=str(dest.with_suffix(".validation.json")),trial=trial["dest"],rows=actual["rows"],
               candidates=actual["candidates"],published_utc=time.time(),content_training=content_training(spec["refs"]),
               content_training_by_branch=scopes(spec["refs"]))
    prev=ledger["packages"].get(ledger["preferred"].get(family))
    alias=None
    if not prev or entry["selection_A"]>prev["selection_A"]:
        alias=SUB/("fusion_"+family+"_prefix.zip")
        validated_copy(dest,alias,"highest A-half AUC among published and B-confirmed packages of this family")
        ledger["preferred"][family]=key
        entry["promoted_as"]=str(alias)
    ledger["packages"][key]=entry
    atomic_json(MANIFEST,ledger)
    atomic_json(R5/"latest_release.json",entry)
    print("READY",entry["zip"],entry["sha256"],flush=True)
    entry["local_notification_requested"]=notify(entry,alias)
    atomic_json(MANIFEST,ledger)

def dependencies(kind):
    result={"dev":False,"shadow":False,"train_plus_dev_test":False}
    try:
        p=branch(kind,"dev");result["dev"]=True;result["selected_dev"]=p.name
    except FileNotFoundError:return result
    try:shadow_path(kind);result["shadow"]=True
    except FileNotFoundError:pass
    try:
        ref=matched_refit(kind,p);result["train_plus_dev_test"]=ref.get("training",["train","dev"])==["train","dev"]
        if not result["train_plus_dev_test"]:result["train_only_test"]=True
    except FileNotFoundError:pass
    return result

def candidates(ledger):
    trials=catalog();evaluated=[];queue=[]
    if frozenset() not in trials:return evaluated,queue
    for k,t in trials.items():
        if not k:continue
        if "e5" in k and not completed("e5"):continue
        if (ROOT/"artifacts/round7/placement.json").exists() and ("qlora" in k or ("fullbge_s2" in k and not completed("fullbge_s2"))):continue
        gate=assess(k,trials)
        row={**t,"gate":gate}
        evaluated.append(row)
        if not gate["release"]:continue
        try:records=refit_records(t)
        except FileNotFoundError:
            row["state"]="waiting_for_matched_test";continue
        key=signature(t,records)
        if key in ledger["packages"]:
            row["state"]="published";continue
        row["state"]="qualified"
        queue.append(dict(key=key,trial=t,gate=gate,refs=records))
    queue.sort(key=lambda x:(x["gate"]["family"]!="v2",-x["trial"]["selection_A"],x["key"]))
    return evaluated,queue

def watch():
    R5.mkdir(parents=True,exist_ok=True);BUILDS.mkdir(parents=True,exist_ok=True)
    with (R5/"publisher.lock").open("a") as service:
        fcntl.flock(service,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ledger=manifest();atomic_json(MANIFEST,ledger)
        active=None;proc=None;pending_errors={}
        while True:
            try:
                if active is not None and proc.poll() is not None:
                    work=BUILDS/active["key"]
                    try:
                        if proc.returncode:raise RuntimeError(f"Builder exited with {proc.returncode}")
                        register(active,work/(stem(active)+".zip"),ledger)
                        pending_errors.pop(active["key"],None)
                    except (OSError,ValueError,RuntimeError,KeyError) as exc:
                        prev=pending_errors.get(active["key"],{}).get("attempts",0)
                        pending_errors[active["key"]]=dict(attempts=prev+1,retry_after=time.time()+min(300,30*(prev+1)),
                            log=str(work/"build.log"),exit_code=proc.returncode,error=repr(exc))
                    active=None;proc=None
                evaluated,queue=candidates(ledger)
                if time.time()>=LOCK:
                    lockfile=ROOT/("artifacts/round7/final_lock.json" if (ROOT/"artifacts/round7/offline_lock.json").exists() else "artifacts/round3/lock.json")
                    if lockfile.is_file():
                        allowed={tuple(t["models"]) for t in read(lockfile)["candidates"]}
                        queue=[q for q in queue if tuple(q["trial"]["models"]) in allowed]
                if proc is None and time.time()<SAFE_END:
                    for item in queue:
                        if time.time()<pending_errors.get(item["key"],{}).get("retry_after",0):continue
                        work=BUILDS/item["key"];work.mkdir(exist_ok=True)
                        cfg=work/"spec.json";atomic_json(cfg,item)
                        with (work/"build.log").open("ab",buffering=0) as log:
                            proc=subprocess.Popen([sys.executable,"-u","scripts/incremental.py","build",str(cfg)],
                                cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL)
                        active=item;break
                running=None
                if active is not None:
                    work=BUILDS/active["key"]
                    progress=work/(stem(active)+".progress.json")
                    if not progress.is_file():progress=work/"worker.json"
                    running=dict(models=active["trial"]["models"],pid=proc.pid,progress=read(progress) if progress.is_file() else {})
                closed=proc is None and (time.time()>=SAFE_END or (time.time()>=LOCK and not queue and lockfile.is_file()))
                state="CLOSED" if closed else "BUILDING" if active else "WATCHING"
                atomic_json(STATUS,dict(stage=state,updated=time.time(),pid=os.getpid(),minimum_full_dev=MIN_DEV,
                    dependencies={k:dependencies(k) for k in EXTRAS},evaluated=evaluated,active=running,
                    eligible_queue=[q["trial"]["models"] for q in queue],published=len(ledger["packages"]),errors=pending_errors))
                if closed:return
            except (OSError,ValueError,RuntimeError,KeyError) as exc:
                atomic_json(STATUS,dict(stage="RETRYING",updated=time.time(),error=repr(exc)))
                print("retry",repr(exc),flush=True)
            time.sleep(15)

def show():
    data=read(STATUS) if STATUS.is_file() else {}
    print(time.strftime("%m-%d %H:%M:%S",time.localtime(data.get("updated",time.time()))),data.get("stage","STARTING"),
          "dev floor",MIN_DEV,"packages",data.get("published",0))
    for kind,v in data.get("dependencies",{}).items():
        print(kind,"dev",v["dev"],"shadow",v["shadow"],"train+dev test",v["train_plus_dev_test"])
    for row in data.get("evaluated",[]):
        g=row["gate"]
        print("+".join(row["models"][3:]),"A %.6f B %.6f full %.6f"%(row["selection_A"],row["heldout_B"],row["full_dev"]),
              "A pass",g["A_pass"],"B pass",g["B_pass"],"delta %+.6f"%g["full_delta_vs_stated"],row.get("state",""))
    if data.get("active"):print("building",data["active"])
    if data.get("error") or data.get("errors"):print("errors",data.get("error") or data["errors"])
    book=manifest()
    for k,e in book["packages"].items():
        print("READY",e["zip"],e["sha256"])
        if book["preferred"].get(e["family"])==k:print("preferred",e["promoted_as"])

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("action",choices=["watch","build","status"])
    parser.add_argument("spec",nargs="?",type=Path);args=parser.parse_args()
    if args.action=="watch":watch()
    elif args.action=="build":build(args.spec)
    else:show()
