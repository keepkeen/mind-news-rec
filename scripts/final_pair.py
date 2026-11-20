"""Follow the published prefix decision and release its fixed-weight forward comparison."""
import argparse
from dataclasses import dataclass
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from zoneinfo import ZoneInfo
from colab_jobs import ART,ROOT,atomic_json
from independent_submission import TEST_SHAPE

PAIR=ART/"final_pair"
SUB=ART/"submissions"
R7=ART/"round7"
RAW=ROOT/"data/large_test/MINDlarge_test/behaviors.tsv"
ZONE=ZoneInfo("Asia/Shanghai")
LOCK=datetime(2026,11,20,4,30,tzinfo=ZONE).timestamp()
DEADLINE=datetime(2026,11,20,8,30,tzinfo=ZONE).timestamp()
SLICES=("all","user_half_fit","user_half_check")

def read(p):
    return json.loads(Path(p).read_text())

def digest(p):
    with Path(p).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def local(t):
    return datetime.fromtimestamp(t,ZONE).strftime("%Y-%m-%d %H:%M:%S %Z")

@dataclass(frozen=True)
class PairJob:
    work: Path
    submissions: Path
    decision: Path
    raw: Path
    shape: tuple
    deadline: float
    daily_groups: int=40000
    dev_limit: int|None=None

    @property
    def rehearsal(self):
        return self.dev_limit is not None

    @property
    def archive(self):
        return self.submissions/"final_forward_experimental.zip"

    @property
    def receipt(self):
        return self.submissions/"FINAL_RECEIPT.json"

    def status(self,stage,**info):
        atomic_json(self.work/"status.json",dict(stage=stage,utc=time.time(),pid=os.getpid(),rehearsal=self.rehearsal,**info))

def layout(rehearsal_dir=None):
    if rehearsal_dir is None:return PairJob(PAIR,SUB,R7/"final_decision.json",RAW,TEST_SHAPE,DEADLINE)
    d=Path(rehearsal_dir).resolve()
    if d.parent!=(PAIR/"rehearsals").resolve():raise ValueError("Invalid rehearsal output directory")
    cfg=read(d/"scope.json")
    return PairJob(d,d/"submissions",d/"final_decision.json",d/"behaviors.tsv",tuple(cfg["test_shape"]),
                   cfg["deadline"],cfg["daily_groups"],cfg["dev_count"])

def check_archive(proof,path,shape):
    if ((proof["rows"],proof["candidates"])!=shape or proof["root_entries"]!=["prediction.txt"] or
        not all(proof[k] for k in ("id_order","two_fields","permutations")) or digest(path)!=proof["sha256"]):
        raise ValueError(f"Independent archive proof does not match {path}")

def test_training(name,record,proof,expected_scope,dev_hash):
    training=proof["training"]
    if ((proof["series"],proof["seed"],proof["epoch"])!=(name,record["seed"],record["epoch"]) or
            training!=record.get("training",["train","dev"]) or training!=expected_scope or
            training not in (["train"],["train","dev"])):
        raise ValueError(f"Official training source mismatch for {name}")
    if training==["train"] and (not proof.get("dev_ranks_match") or proof.get("selected_dev_sha256")!=dev_hash):
        raise ValueError(f"Train-only alternative does not match the selected dev checkpoint for {name}")
    return training

def pinned_inputs(decision,job):
    trial=decision["selection"];models=trial["models"];weights=trial["weights"]
    source=Path(trial["dest"]);official=Path(decision["zip"])
    official_meta=read(official.with_suffix(".validation.json"))
    policy=read(source/"candidate_policy.json")
    packaged_policy=read(official_meta["policy"])
    if (official_meta["future_exposure"] is not False or official_meta["tree_fit_days"]!="Nov 12–15" or
        official_meta["sha256"]!=decision["sha256"] or decision["independent"]["sha256"]!=decision["sha256"] or
        decision["models"]!=models or decision["weights"]!=weights or len(weights)!=len(models)+1 or
        not all(p["type"]=="prefix-only" and p["models"]==models and p["weights"]==weights and
                p["columns"]==policy["columns"] and p["base_files"]==policy["base_files"] for p in (policy,packaged_policy))):
        raise ValueError("Official archive, fitted policy and locked weights disagree")
    check_archive(decision["independent"],official,job.shape)
    records=trial["matched_refit"]
    if (set(records)!=set(models) or official_meta["matched_refit"]!=records or
        official_meta["components"]!=[records[n]["test"] for n in models]):
        raise ValueError("The official content inputs have changed")
    branches={}
    for name,base_file in zip(models,policy["base_files"],strict=True):
        rec=records[name];oof=trial["oof_inputs"][name]
        proof=read(rec["receipt"])
        training=test_training(name,rec,proof,decision["content_training_by_branch"][name],oof["dev"]["sha256"])
        if Path(rec["eval_file"]).resolve()!=Path(base_file).resolve() or Path(oof["dev"]["path"]).resolve()!=Path(base_file).resolve():
            raise ValueError(f"Official scoring provenance mismatch for {name}")
        cp=Path(rec["receipt"]).parent/proof["checkpoint"]["name"];test=Path(rec["test"])
        if (proof["test_scores"]["name"]!=test.name or proof["test_scores"]["sha256"]!=rec["test_sha256"]
                or proof["checkpoint"]["sha256"]!=rec["checkpoint_sha256"]):
            raise ValueError(f"Test score or checkpoint receipt mismatch for {name}")
        files=dict(dev=oof["dev"]["path"],shadow=oof["shadow"]["path"],test=str(test),checkpoint=str(cp))
        hashes=dict(dev=oof["dev"]["sha256"],shadow=oof["shadow"]["sha256"],test=rec["test_sha256"],checkpoint=rec["checkpoint_sha256"])
        for role,f in files.items():
            if digest(f)!=hashes[role]:raise ValueError(f"Changed {role} input for {name}: {f}")
        branches[name]=dict(seed=rec["seed"],epoch=rec["epoch"],training=training,files=files,hashes=hashes,
                            receipt=rec["receipt"])
    audit=dict(models=models,weights=weights,columns=policy["columns"],branches=branches,gate=trial["gate"],
               final_sha256=decision["sha256"],official_zip=str(official),trial=str(source),
               dev_tree_sha256=digest(source/"temporal_tree.lgb"),forward_definition_sha256=digest(ROOT/"scripts/experimental_exposure.py"),
               daily_groups=job.daily_groups,dev_limit=job.dev_limit,test_shape=list(job.shape))
    audit["input_signature"]=fingerprint(audit)
    return audit

def result_receipt(job,decision,audit,build):
    from experimental_exposure import FORWARD
    exp=read(build/"independent_validation.json");model=read(build/"policy.json");metrics=read(build/"dev_metrics.json")
    weights=decision["weights"];names=decision["models"];branches=audit["branches"];scope=decision["content_training_by_branch"]
    if (exp["official_sha256"]!=decision["sha256"] or exp["input_signature"]!=audit["input_signature"] or not exp["future_exposure"]
            or exp["models"]!=names or exp["weights"]!=weights or exp["content_training_by_branch"]!=scope
            or model["models"]!=names or model["weights"]!=weights or model["weight_search_performed"] is not False
            or model["input_signature"]!=audit["input_signature"] or metrics["selection"]["package_weights"]!=weights
            or model["columns"]!=audit["columns"]+list(FORWARD)):
        raise ValueError("Forward archive did not use the published branch order, weights and test inputs")
    check_archive(exp,job.archive,job.shape)
    check_archive(decision["independent"],decision["zip"],job.shape)
    actual=read(job.decision)
    if actual["sha256"]!=decision["sha256"] or actual["selection"]!=decision["selection"]:
        raise ValueError("The official decision changed while building its pair")
    out={}
    trial=decision["selection"]
    for key,field in zip(SLICES,("full_dev","selection_A","heldout_B"),strict=True):
        a=metrics["formal"][key];b=metrics["experimental_fixed"][key]
        diff=b["AUC"]-a["AUC"]
        if (a["impressions"]!=b["impressions"] or abs(diff-metrics["delta"][key])>1e-12 or
                abs(a["AUC"]-trial[field])>1e-11):
            raise ValueError(f"Official/experimental dev scope mismatch for {key}")
        out[key]=dict(impressions=a["impressions"],official=a["AUC"],experimental=b["AUC"],delta=diff)
    training=read(build/"training.json")
    reference_dir=Path(trial["dest"]) if job.rehearsal else ART/"experimental_forward"
    reference=read(reference_dir/"training.json")
    for field in ("fit_days","final_tree_days","groups_per_day","fit_groups","final_fit_groups","train_rows_sha256","dev_refit_rows_sha256"):
        if training[field]!=reference[field]:raise ValueError(f"Training rows differ from the paired reference: {field}")
    if not job.rehearsal and read(build/"coverage.json")!=read(reference_dir/"coverage.json"):
        raise ValueError("Exposure window coverage differs from the R9 reference")
    # Both archives are already independently validated and published before this atomic receipt appears.
    finished=time.time()
    return dict(status="READY",rehearsal=job.rehearsal,completed_utc=finished,completed_beijing=local(finished),
        deadline_beijing=local(job.deadline),on_time=finished<=job.deadline,
        official=dict(role="official_strict_prefix",zip=decision["zip"],sha256=decision["sha256"],
                      future_exposure=False,decision_utc=decision["utc"],independent_validation=decision["independent"]),
        experimental=dict(role="experimental_forward",zip=str(job.archive),sha256=exp["sha256"],
                          future_exposure=True,independent_validation=exp,added_features=model["columns"][len(audit["columns"]):]),
        dev_auc=out,models=names,weight_order=names+["LightGBM"],weights=weights,
        content_training_by_branch=scope,
        branches={n:dict(seed=branches[n]["seed"],epoch=branches[n]["epoch"],
                         test_training=scope[n],checkpoint_sha256=branches[n]["hashes"]["checkpoint"],
                         test_scores_sha256=branches[n]["hashes"]["test"]) for n in names},
        shared_input_signature=audit["input_signature"],input_manifest=str(build/"inputs.json"),
        training=training,dev_metrics=str(build/"dev_metrics.json"))

def build_pair(job):
    from experimental_forward import ForwardRun,run_forward
    decision=read(job.decision)
    audit=pinned_inputs(decision,job)
    build=job.work/"builds"/audit["input_signature"]
    build.mkdir(parents=True,exist_ok=True)
    with (build/"build.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if job.receipt.is_file():
            done=read(job.receipt)
            if (done.get("status")=="READY" and done.get("shared_input_signature")==audit["input_signature"]
                    and done["official"]["sha256"]==decision["sha256"] and job.archive.is_file()
                    and digest(job.archive)==done["experimental"]["sha256"]):
                job.status("ready",receipt=str(job.receipt),official_sha256=done["official"]["sha256"],
                           experimental_sha256=done["experimental"]["sha256"],dev_auc=done["dev_auc"])
                return done
        audit_path=build/"inputs.json";atomic_json(audit_path,audit)
        job.status("building_forward",official=decision["zip"],official_sha256=decision["sha256"],progress=str(build/"progress.json"))
        receipt=build/"independent_validation.json"
        ready=(receipt.is_file() and job.archive.is_file() and read(receipt).get("input_signature")==audit["input_signature"]
               and read(receipt)["sha256"]==digest(job.archive))
        if not ready:
            run=ForwardRun(work=build,archive=job.archive,formal=Path(decision["zip"]),raw=job.raw,test_shape=job.shape,
                           daily_groups=job.daily_groups,dev_limit=job.dev_limit,locked_weights=True)
            run_forward(run,audit_path,Path(audit["trial"]),resume=True)
        result=result_receipt(job,decision,audit,build)
        atomic_json(job.receipt,result)
        job.status("ready",receipt=str(job.receipt),official_sha256=result["official"]["sha256"],experimental_sha256=result["experimental"]["sha256"],
                   dev_auc=result["dev_auc"])
        print("READY",json.dumps({key:result[key] for key in ("completed_beijing","dev_auc")},ensure_ascii=False),
              result["official"]["zip"],result["official"]["sha256"],result["experimental"]["zip"],result["experimental"]["sha256"],flush=True)
        return result

def notify(result):
    delta=result["dev_auc"]["all"]["delta"]
    body=(f"正式 {result['official']['sha256'][:12]}  实验 {result['experimental']['sha256'][:12]}  "
          f"完整 dev 增量 {delta:+.6f}\n{SUB/'FINAL_RECEIPT.json'}")
    script='on run argv\n display notification (item 1 of argv) with title "MIND 两个最终包已校验"\nend run'
    try:subprocess.run(["/usr/bin/osascript","-e",script,body],capture_output=True,timeout=8)
    except (OSError,subprocess.TimeoutExpired):pass

def watch(job):
    job.work.mkdir(parents=True,exist_ok=True)
    with (job.work/"watch.lock").open("a") as gate:
        try:fcntl.flock(gate,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:return
        job.status("waiting_for_official",decision=str(job.decision),scheduled_lock_beijing=local(LOCK),deadline_beijing=local(job.deadline))
        next_recovery=LOCK+60;attempt=0
        while time.time()<job.deadline:
            if not job.decision.is_file():
                if not job.rehearsal and time.time()>=next_recovery:
                    # The official coordinator and this recovery entry point share one publication lock.
                    next_recovery=time.time()+90
                    try:subprocess.run([sys.executable,"scripts/recovery_final.py"],cwd=ROOT,
                                       timeout=max(1,job.deadline-time.time()-20*60),check=True)
                    except (OSError,subprocess.SubprocessError) as exc:print("official completion check",repr(exc),flush=True)
                time.sleep(0.25 if job.rehearsal else 5);continue
            attempt+=1
            cmd=["/usr/bin/nice","-n","12",sys.executable,"-u","scripts/final_pair.py","build"]
            if job.rehearsal:cmd+=["--rehearsal-dir",str(job.work)]
            try:
                remaining=max(1,job.deadline-time.time())
                attempt_limit=min(35*60,remaining)
                if attempt==1 and remaining>=28*60:attempt_limit=min(attempt_limit,remaining-14*60)
                with (job.work/"build.log").open("a") as log:
                    subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                                   env=dict(os.environ,OPENBLAS_NUM_THREADS="1",VECLIB_MAXIMUM_THREADS="1",OMP_NUM_THREADS="4"),
                                   check=True,timeout=attempt_limit)
                result=read(job.receipt)
                if not job.rehearsal:notify(result)
                return
            except (OSError,ValueError,subprocess.SubprocessError) as exc:
                job.status("retrying_forward",attempt=attempt,error=repr(exc),log=str(job.work/"build.log"))
                if job.rehearsal:raise
                time.sleep(min(60,10*attempt,max(0,job.deadline-time.time())))
        job.status("deadline_missed",receipt=str(job.receipt),log=str(job.work/"build.log"))
        raise TimeoutError("The two verified archives are not ready by the deadline")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=("arm","watch","build","status"))
    p.add_argument("--rehearsal-dir",type=Path)
    args=p.parse_args();job=layout(args.rehearsal_dir)
    if args.action=="arm":
        from recovery_final import arm_incumbent
        if args.rehearsal_dir or time.time()>=LOCK:raise ValueError("Arming is only for the production setup before lock")
        floor=arm_incumbent()
        print("incumbent",floor["selection"]["full_dev"],floor["sha256"],floor["zip"])
    elif args.action=="status":
        print(json.dumps(read(job.work/"status.json"),ensure_ascii=False,indent=2))
    elif args.action=="build":build_pair(job)
    else:watch(job)

if __name__=="__main__":main()
