"""Select a completed prefix-only pipeline at the lock, refit its tree, publish a verified final."""
from pathlib import Path
import fcntl
import subprocess
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from colab_jobs import ART,ROOT,atomic_json
from fusion import refit_tree,score_test
from incremental_policy import BASE,EXTRAS,MANIFEST,assess,catalog,completed,dev_ready,planned,read,refit_records,registered,trial_info,validated_copy

THREE=list(BASE)
DIR=ART/"round3";PROGRESS=ART/"local/coordinator.json"
THREE_DIR=ART/"fusion"
TODAY=ART/"submissions/fusion_three_train_prefix.zip"
BASELINE=ART/"submissions/baseline_train.zip"
ZONE=ZoneInfo("Asia/Shanghai")
LOCK=datetime(2026,11,20,4,30,tzinfo=ZONE).timestamp()
SAFE_END=datetime(2026,11,20,8,5,tzinfo=ZONE).timestamp()

def status(stage,**info):atomic_json(PROGRESS,dict(stage=stage,time=time.time(),**info))
def eligible_trials(require_refit=True):
    trials=catalog()
    if frozenset() not in trials:return []
    results=[]
    for key,item in trials.items():
        recovered=(ART/"round7/placement.json").is_file()
        if "e5" in key and not (completed("e5") if require_refit else dev_ready("e5")):continue
        if recovered and ("qlora" in key or ("fullbge_s2" in key and not completed("fullbge_s2"))):continue
        decision=assess(key,trials)
        if not decision["A_pass"] or not decision["B_pass"]:continue
        if require_refit:
            try:item["matched_refit"]=refit_records(item)
            except FileNotFoundError:continue
        item["gate"]=decision
        results.append(item)
    return sorted(results,key=lambda t:(-t["selection_A"],len(t["models"]),t["models"]))

def atomic_publish(source,reason):
    return validated_copy(source,ART/"submissions/final_prefix.zip",reason)

def prepare():
    DIR.mkdir(parents=True,exist_ok=True)
    attempts={}
    times=[datetime(2026,11,19,h,m,tzinfo=ZONE) for h,m in ((15,35),(17,25),(18,35),(20,20),(23,0),(23,35))]
    while time.time()<LOCK:
        recovered=(ART/"round7/placement.json").is_file()
        if recovered:
            from recovery_status import snapshot
            for hour,minute in ((0,5),(3,25),(4,15)):
                at=datetime(2026,11,20,hour,minute,tzinfo=ZONE)
                path=ART/"round7"/("check_"+at.strftime("%H%M")+".json")
                if time.time()>=at.timestamp() and not path.exists():atomic_json(path,snapshot())
        for at in times:
            out=DIR/("check_"+at.strftime("%H%M")+".json")
            if time.time()>=at.timestamp() and not out.exists():
                from fleet_status import summary
                atomic_json(out,summary())
        ready=[k for k in EXTRAS if (not recovered or k!="qlora") and dev_ready(k)]
        existing=catalog()
        ran=False
        ranked=eligible_trials(require_refit=False) if "e5" in ready or (recovered and "fullbge_s2" in ready) else []
        incumbent=ranked[0]["models"] if ranked else []
        for combo in planned(ready,incumbent):
            if frozenset(combo) in existing:continue
            dest=DIR/("with_"+"_".join(combo))
            old=attempts.get(dest,(0,0))
            if old[1]>time.time() or time.time()>=LOCK-15:continue
            dest.mkdir(parents=True,exist_ok=True)
            names=THREE+list(combo)
            status("validating_stronger_content",models=names,dest=str(dest),lock_epoch=LOCK)
            with (dest/"worker.log").open("a") as log:
                try:
                    subprocess.run([sys.executable,"scripts/fusion.py","dev","--models",",".join(names),
                                    "--dest",str(dest),"--compact"],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   timeout=max(1,LOCK-time.time()-5),check=True)
                except (subprocess.SubprocessError,OSError) as exc:
                    n=old[0]+1;attempts[dest]=(n,time.time()+min(600,30*n))
                    atomic_json(dest/"attempt_error.json",dict(error=repr(exc),attempt=n))
            ran=True
            break
        if ran:continue
        pool=eligible_trials()
        dev_pool=eligible_trials(require_refit=False)
        status("awaiting_0430_selection",complete_new_branches=ready,best_matched=pool[0] if pool else None,
               best_evaluated=dev_pool[0] if dev_pool else None,
               today_zip=str(TODAY),lock_epoch=LOCK)
        time.sleep(min(25,max(.1,LOCK-time.time())))

def final_attempt(dest):
    from incremental_policy import sha256
    with (dest/"official_attempt.lock").open("a") as guard:
        fcntl.flock(guard,fcntl.LOCK_EX)
        pinned=None
        lock=ART/"round7/final_lock.json"
        if lock.is_file():
            pinned=next(t["matched_refit"] for t in read(lock)["candidates"] if Path(t["dest"]).resolve()==dest.resolve())
        output="locked_"+dest.name
        staged=DIR/"staging"/(output+".zip")
        receipt=staged.with_suffix(".validation.json")
        if pinned and receipt.is_file() and staged.is_file():
            existing=read(receipt)
            if existing["matched_refit"]==pinned and existing["sha256"]==sha256(staged):return
        pol=read(dest/"candidate_policy.json")
        atomic_json(dest/"policy.json",pol)
        refit_tree(dest)
        score_test(dest,"policy.json","ranker.lgb",output,refit_base=True,output_dir=DIR/"staging",pinned_records=pinned)

def main():
    prepare()
    if (ART/"round7/offline_lock.json").is_file():
        from recovery_final import at_lock
        return at_lock()
    lockfile=DIR/"lock.json"
    if not lockfile.exists():
        atomic_json(lockfile,dict(locked_utc=time.time(),candidates=eligible_trials(),
                content="train+dev seed-matched checkpoint for each branch, selected by epoch on its train-only dev curve",
                final_tree="40k full impressions per day, Nov 12, 13, 14, 15, out-of-training base logits",
                rule="maximum selection-A AUC among completed trials that also passed B relative to the three-model incumbent"))
    lock=read(lockfile)
    for candidate in lock["candidates"]:
        try:already=registered(candidate,candidate["matched_refit"])
        except (OSError,ValueError,KeyError) as exc:
            status("checking_locked_source",error=repr(exc));already=None
        if already:
            try:
                receipt=atomic_publish(already,"locked dev-selected policy, reusing its validated four-day-tree package")
                status("DONE",zip=receipt["zip"],sha256=receipt["sha256"],models=candidate["models"])
                return
            except (OSError,ValueError) as exc:status("rebuilding_locked_archive",error=str(exc))
        remaining=SAFE_END-time.time()
        if remaining<300:break
        d=Path(candidate["dest"])
        status("building_final_prefix_zip",trial=candidate)
        try:
            # Timebox a failed branch so there is always time to publish the verified fallback.
            with (d/"final_build.log").open("a") as log:
                subprocess.run([sys.executable,"scripts/coordinate.py","attempt",str(d)],cwd=ROOT,
                               stdout=log,stderr=subprocess.STDOUT,timeout=min(remaining,3900),check=True)
            source=DIR/"staging"/("locked_"+d.name+".zip")
            receipt=atomic_publish(source,"highest available locked policy with a successfully rebuilt tree")
            status("DONE",zip=receipt["zip"],sha256=receipt["sha256"],models=candidate["models"])
            return
        except Exception as e:
            atomic_json(d/"final_failure.json",dict(error=repr(e),time=time.time()))
    published=[]
    if MANIFEST.is_file():
        locked={tuple(t["models"]) for t in lock["candidates"]}
        entries=sorted(read(MANIFEST)["packages"].values(),key=lambda e:e["selection_A"],reverse=True)
        published=[Path(e["zip"]) for e in entries if tuple(e["models"]) in locked]
    for backup in (*published,TODAY,BASELINE):
        if backup.is_file() and backup.with_suffix(".validation.json").is_file():
            try:receipt=atomic_publish(backup,"deadline or failed final rebuild; preserved validated candidate")
            except (OSError,ValueError):continue
            status("DONE_WITH_FALLBACK",zip=receipt["zip"],sha256=receipt["sha256"],source=str(backup))
            return
    raise RuntimeError("No verified candidate on disk")

if __name__=="__main__":
    if len(sys.argv)>1 and sys.argv[1]=="attempt":final_attempt(Path(sys.argv[2]))
    else:
        try:main()
        except Exception as exc:status("FAILED",error=repr(exc),validated_fallback=str(TODAY));raise

