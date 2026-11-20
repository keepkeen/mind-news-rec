"""Publish the 04:30 prefix decision independently of the subsequent forward experiment."""
import fcntl
from pathlib import Path
import subprocess
import sys
import time
from colab_jobs import ART,ROOT,atomic_json
from coordinate import eligible_trials,SAFE_END,LOCK,status
from e5_rescue_policy import content_training,scopes
from fusion import shadow_path
from incremental_policy import read,registered,validated_copy,sha256
from independent_submission import TEST_SHAPE,verify

STAGE=ART/"round7"
PAIR=ART/"final_pair"
SUB=ART/"submissions"
RAW=ROOT/"data/large_test/MINDlarge_test/behaviors.tsv"
OFFICIAL_BY=SAFE_END-600  # 07:55 Beijing, with 35 minutes left for the paired CPU job.

def pin(trial):
    """Pin the exact OOF paths and hashes in addition to the epoch-matched test receipts."""
    if "oof_inputs" in trial:return trial
    policy=read(Path(trial["dest"])/"candidate_policy.json")
    if policy["type"]!="prefix-only" or (trial["models"],trial["weights"])!=(policy["models"],policy["weights"]):
        raise ValueError("Trial differs from the validated prefix policy")
    oof={}
    for name,path in zip(trial["models"],policy["base_files"],strict=True):
        dev=Path(path);shadow=shadow_path(name)
        if dev.resolve()!=Path(trial["matched_refit"][name]["eval_file"]).resolve():raise ValueError(name)
        oof[name]={k:dict(path=str(p),sha256=Path(str(p)+".verified").read_text().strip())
                   for k,p in (("dev",dev),("shadow",shadow))}
    return dict(trial,oof_inputs=oof)

def original():
    trial=pin(read(STAGE/"offline_lock.json")["selected"])
    signed=read(STAGE/"independent_validation.json")
    return dict(zip=str(SUB/"final_prefix.zip"),sha256=signed["sha256"],selection=trial)

def best_published(pool, floor):
    """Prefer the highest verified, A-selected and B-confirmed immutable archive."""
    for t in pool:
        if t["selection_A"]<=floor["selection"]["selection_A"]:break
        if not t["gate"]["release"]:continue
        try:
            source=registered(t,t["matched_refit"])
            if not source:continue
            receipt=read(source.with_suffix(".validation.json"))
            if (receipt.get("future_exposure") is False and receipt.get("tree_fit_days")=="Nov 12–15"
                    and receipt.get("matched_refit")==t["matched_refit"] and sha256(source)==receipt["sha256"]):
                return dict(zip=str(source),sha256=receipt["sha256"],selection=pin(t))
        except (OSError,ValueError,KeyError) as exc:
            atomic_json(Path(t["dest"])/"official_source_issue.json",dict(utc=time.time(),error=repr(exc)))
    return floor

def arm_incumbent():
    path=PAIR/"incumbent.json"
    floor=read(path) if path.is_file() else original()
    floor=best_published(eligible_trials(),floor)
    if sha256(floor["zip"])!=floor["sha256"]:raise ValueError("Published incumbent hash changed")
    atomic_json(path,floor)
    return floor

def publish_official(trial,source,target,decision_file,raw=RAW,shape=TEST_SHAPE,reason="locked"):
    """The atomic decision is the durable signal for the separate, CPU-only pair worker."""
    source,target=Path(source),Path(target)
    meta=read(source.with_suffix(".validation.json"))
    records=trial["matched_refit"]
    if (meta["matched_refit"]!=records or meta["future_exposure"] is not False
            or meta["tree_fit_days"]!="Nov 12–15" or meta["content_training"]!=content_training(records)):
        raise ValueError("Prefix archive does not match the locked sources")
    validation=verify(source,raw,expected_shape=shape)
    if validation["sha256"]!=meta["sha256"]:raise ValueError("Official validation digest changed")
    validation["zip"]=str(target)
    if source.resolve()!=target.resolve():
        validated_copy(source,target,reason,expected_shape=shape,independent=validation,selection=trial,
                       models=trial["models"],weights=trial["weights"],content_training_by_branch=scopes(records))
    result=dict(result=reason,zip=str(target),sha256=validation["sha256"],utc=time.time(),
                models=trial["models"],weights=trial["weights"],selection=trial,
                content_training=content_training(records),content_training_by_branch=scopes(records),
                independent=validation)
    if shape==TEST_SHAPE:
        result.update(immutable_backup=str(SUB/"final_prefix.zip"),
                      backup_sha256=read(STAGE/"independent_validation.json")["sha256"])
    atomic_json(decision_file,result)
    return result

def select_and_publish():
    decision_file=STAGE/"final_decision.json"
    if decision_file.is_file():
        existing=read(decision_file)
        if Path(existing["zip"]).is_file() and sha256(existing["zip"])==existing["sha256"]:return existing
    lockfile=STAGE/"final_lock.json"
    if not lockfile.is_file():
        floor=read(PAIR/"incumbent.json") if (PAIR/"incumbent.json").is_file() else original()
        try:
            pool=[t for t in eligible_trials() if all((Path(t["dest"])/n).stat().st_mtime<LOCK for n in
                  ("candidate_policy.json","dev_metrics.json","weights_from_A.json","heldout_B.json","temporal_tree.lgb"))]
        except (OSError,ValueError,KeyError) as exc:
            atomic_json(STAGE/"official_lock_scan_issue.json",dict(utc=time.time(),error=repr(exc)))
            pool=[]
        floor=best_published(pool,floor)
        better=[]
        for t in pool:
            if not t["gate"]["release"] or t["selection_A"]<=floor["selection"]["selection_A"]:continue
            try:better.append(pin(t))
            except (OSError,ValueError,KeyError) as exc:
                atomic_json(Path(t["dest"])/"official_source_issue.json",dict(utc=time.time(),error=repr(exc)))
        atomic_json(lockfile,dict(locked_utc=time.time(),scheduled_utc=LOCK,incumbent=floor,
                    candidates=better+[floor["selection"]],fallback_sha256=floor["sha256"],
                    rule="highest eligible user-half-A AUC; B confirmation; verified incumbent floor"))
    locked=read(lockfile);floor=locked["incumbent"]
    for trial in locked["candidates"]:
        if trial["selection_A"]<=floor["selection"]["selection_A"] or time.time()>=OFFICIAL_BY-90:break
        d=Path(trial["dest"])
        try:
            source=registered(trial,trial["matched_refit"])
            if source is None:
                with (d/"final_recovery.log").open("a") as log:
                    subprocess.run([sys.executable,"scripts/coordinate.py","attempt",str(d)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   timeout=min(3300,max(1,OFFICIAL_BY-time.time()-90)),check=True)
                source=ART/"round3/staging"/("locked_"+d.name+".zip")
            result=publish_official(trial,source,SUB/"final_v2_prefix.zip",decision_file,reason="improved")
            status("DONE",zip=result["zip"],sha256=result["sha256"]);return result
        except (OSError,ValueError,subprocess.SubprocessError,KeyError) as exc:
            atomic_json(d/"r7_build_failure.json",dict(error=repr(exc),time=time.time()))
    source=Path(floor["zip"])
    if sha256(source)!=floor["sha256"]:raise ValueError("Locked incumbent archive failed checksum")
    trial=floor["selection"]
    target=SUB/"final_v2_prefix.zip" if {"fullbge_s2","e5"}.intersection(trial["models"]) else source
    result=publish_official(trial,source,target,decision_file,reason="best_validated_incumbent")
    status("DONE_WITH_VERIFIED_INCUMBENT",zip=result["zip"],sha256=result["sha256"]);return result

def at_lock(block=True):
    if time.time()<LOCK:raise RuntimeError("The official lock time has not arrived")
    with (STAGE/"official_publish.lock").open("a") as f:
        try:fcntl.flock(f,fcntl.LOCK_EX if block else fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return None
        return select_and_publish()

if __name__=="__main__":at_lock(block=False)
