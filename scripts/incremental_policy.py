"""Use the complete user half A to select models; confirm improvements on the other users."""
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import shutil
import time
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np
from colab_jobs import ART,atomic_json
from fusion import branch,matched_refit,shadow_path

BASE=("bge","upper","qwen")
UPPER=("upper_s2","upper_s3")
LATER=("fullbge_s2","asym4","qlora","e5")
EXTRAS=("fullbge",*UPPER,*LATER)
STATED_BASELINE=0.7244
MIN_DEV=0.7264
TRIALS=ART/"round3"
R5=ART/"round5"
SUB=ART/"submissions"
MANIFEST=SUB/"incremental_manifest.json"
TEST_SHAPE=(2370727,93115001)
LOCK_AT=datetime(2026,11,20,4,30,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()

def read(path):
    return json.loads(Path(path).read_text())

def dev_ready(kind):
    if kind!="e5" or not (ART/"round7/e5_fit_incident.json").is_file():return completed(kind)
    from e5_rescue_policy import ready_dev
    try:return ready_dev(branch(kind,"dev"),shadow_path(kind))
    except FileNotFoundError:return False

def completed(kind):
    try:
        p=branch(kind,"dev")
        shadow_path(kind)
    except FileNotFoundError:return False
    hb=p.parent/"heartbeat.json"
    state=read(hb)["status"].get(kind+".status.json",{}) if hb.exists() else {}
    recovery=ART/"round7"
    choice=recovery/"bge_epoch_choice.json"
    if kind=="e5" and (recovery/"e5_fit_incident.json").is_file():
        if not dev_ready(kind):return False
        try:ref=matched_refit(kind,p)
        except FileNotFoundError:return False
        source="refit" if ref["training"]==["train","dev"] else "trainonly"
        out=recovery/("e5_"+source+"_ready_before_lock.json")
        if not out.exists():
            atomic_json(out,dict(selection=ref,lock_utc=LOCK_AT,early_shadow=str(shadow_path(kind)),
                                received_utc=Path(ref["test"]+".verified").stat().st_mtime))
        return True
    if kind=="fullbge_s2" and choice.is_file():
        pinned=read(choice)
        try:
            ref=matched_refit(kind,p);sh=shadow_path(kind);sm=read(sh.parent/(kind+"_shadow_meta.json"))
            tests=Path(ref["test"])
            checks=[p,sh,tests,Path(pinned["eval_checkpoint"]),Path(pinned["refit_checkpoint"])]
            if ref["epoch"]!=2 or ref["seed"]!=1163 or sm.get("epoch")!=2 or sm.get("seed")!=1163 or sm.get("fit_before")!="2019-11-12":return False
            if str(p.resolve())!=pinned["eval_dev"] or not tests.is_file():return False
            if np.load(tests,mmap_mode="r").shape!=(TEST_SHAPE[1],) or np.load(sh,mmap_mode="r").shape!=(int(np.load(ART/"compact/train/offset.npy",mmap_mode="r")[-1]),):return False
            received=max(f.with_name(f.name+".verified").stat().st_mtime for f in checks)
            if received>=LOCK_AT:return False
            marker=recovery/"bge_ready_before_lock.json"
            if not marker.is_file():atomic_json(marker,dict(received_utc=received,lock_utc=LOCK_AT,selection=ref,early_shadow=str(sh)))
            return True
        except FileNotFoundError:return False
    if state.get("phase")!="DONE":return False
    if kind!="e5":return True
    try:
        ref=matched_refit(kind,p)
        shadow=shadow_path(kind)
        fitdir=Path(ref["receipt"]).parent
        cp_files=[d/f"e5{suffix}_e{ep}.pt" for d,suffix in ((p.parent,""),(fitdir,"_refit")) for ep in (1,2,3)]
        data=[(p,"dev"),(shadow,"train"),(Path(ref["test"]),"test")]
        for f,split in data:
            count=int(np.load(ART/"compact"/split/"offset.npy",mmap_mode="r")[-1])
            if np.load(f,mmap_mode="r").shape!=(count,):return False
        verified=[f.with_name(f.name+".verified") for f in cp_files+[v[0] for v in data]]
        if not all(f.is_file() and f.stat().st_size>0 for f in cp_files):return False
        received=max([state["utc_epoch"]]+[v.stat().st_mtime for v in verified])
    except FileNotFoundError:return False
    if received>=LOCK_AT:return False
    receipt=(recovery/"e5_ready_before_lock.json") if (recovery/"placement.json").exists() else ART/"round6/ready_before_lock.json"
    if not receipt.exists():
        atomic_json(receipt,dict(received_utc=received,lock_utc=LOCK_AT,selection=ref,shadow=str(shadow),
            checkpoints=list(map(str,cp_files)),file_hashes={f.name:f.read_text().strip() for f in verified}))
    return True

def trial_info(dest):
    dest=Path(dest)
    if not all((dest/f).is_file() for f in ("candidate_policy.json","weights_from_A.json","heldout_B.json","dev_metrics.json")):return None
    policy=read(dest/"candidate_policy.json")
    report=read(dest/"dev_metrics.json")["blend"]
    names=policy["models"]
    if tuple(names[:3])!=BASE or len(set(names))!=len(names) or not set(names[3:])<=set(EXTRAS):return None
    if policy.get("type")!="prefix-only":raise ValueError(f"Unexpected feature policy in {dest}")
    a,b,full=(report[x] for x in ("user_half_fit","user_half_check","all"))
    if (a["impressions"],b["impressions"],full["impressions"])!=(187811,188660,376471):raise ValueError(f"Incomplete validation {dest}")
    if not all(math.isfinite(x["AUC"]) for x in (a,b,full)):raise ValueError(dest)
    return dict(dest=str(dest),models=names,weights=policy["weights"],selection_A=a["AUC"],heldout_B=b["AUC"],
                full_dev=full["AUC"],weight_search_A=read(dest/"weights_from_A.json")["auc"])

def model_key(models):
    return frozenset(models)-set(BASE)

def catalog():
    out={}
    for d in [ART/"fusion",*sorted(TRIALS.glob("with_*"))]:
        trial=trial_info(d)
        if trial:out[model_key(trial["models"])]=trial
    return out

def ordered_subsets(items):
    return [subset for n in range(len(items)+1) for subset in itertools.combinations(items,n)]

def planned(ready,preferred=()):
    if "fullbge" not in ready:return []
    seeds=tuple(k for k in UPPER if k in ready)
    later=tuple(k for k in LATER if k in ready)
    # First compare the four v2 variants. Before a combination, evaluate each of its removals.
    all_combos=[("fullbge",*s,*extra) for extra in ordered_subsets(later) for s in ordered_subsets(seeds)]
    recovered=bool((ART/"round7/placement.json").exists())
    if "e5" not in ready and not (recovered and "fullbge_s2" in ready):return all_combos
    # Prioritize the incumbent with E5 and its one-branch removals ahead of the remaining Cartesian grid.
    additions=[k for k in (("e5","fullbge_s2") if recovered else ("e5",)) if k in ready]
    next_sets=[]
    for add in additions+([tuple(additions)] if len(additions)>1 else []):
        group={add} if isinstance(add,str) else set(add)
        target=(set(preferred)&set(ready))|{"fullbge"}|group
        next_sets+=[{"fullbge"}|group]+[target-{k} for k in sorted(target-{"fullbge"})]+[target]
    head=[combo for k in next_sets for combo in all_combos if set(combo)==k]
    return list(dict.fromkeys(head+all_combos))

def assess(key,all_trials):
    t=all_trials[key];base=all_trials[frozenset()]
    optional=key-{"fullbge"}
    parents={key-{k} for k in optional}
    if key:parents.add(frozenset())
    if "fullbge" in key and optional:parents.add(frozenset(["fullbge"]))
    missing=[sorted(p) for p in parents if p not in all_trials]
    deltas={"+".join(sorted(p)) or "submitted":t["selection_A"]-all_trials[p]["selection_A"] for p in parents if p in all_trials}
    passes_a=not missing and all(d>0 for d in deltas.values())
    passes_b=t["heldout_B"]>=base["heldout_B"]
    family="v3" if key.intersection(LATER) else "v2" if "fullbge" in key else "initial"
    return dict(family=family,A_pass=passes_a,B_pass=passes_b,missing_A_ablations=missing,A_delta_vs_removals=deltas,
                B_reference=base["heldout_B"],B_delta=t["heldout_B"]-base["heldout_B"],
                full_delta_vs_exact=t["full_dev"]-base["full_dev"],full_delta_vs_stated=t["full_dev"]-STATED_BASELINE,
                full_pass=t["full_dev"]>=MIN_DEV,release=bool(key) and passes_a and passes_b and t["full_dev"]>=MIN_DEV)

def refit_records(trial):
    policy=read(Path(trial["dest"])/"candidate_policy.json")
    return {n:matched_refit(n,p) for n,p in zip(policy["models"],policy["base_files"],strict=True)}

def signature(trial,records):
    policy=read(Path(trial["dest"])/"candidate_policy.json")
    training={}
    for k,p in zip(policy["models"],policy["base_files"],strict=True):
        s=shadow_path(k)
        training[k]=[s.with_name(s.name+".verified").read_text().strip(),Path(p+".verified").read_text().strip()]
    payload=dict(models=policy["models"],weights=policy["weights"],columns=policy["columns"],oof_hashes=training,
                 refits={k:(v["seed"],v["epoch"],v["checkpoint_sha256"],v["test_sha256"]) for k,v in records.items()},
                 tree="lambdarank-100-prefix-v1;40000 complete groups per day 2019-11-12 through 2019-11-15")
    if any(v.get("training")==["train"] for v in records.values()):
        payload["content_training_by_branch"]={k:v.get("training",["train","dev"]) for k,v in records.items()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def sha256(path):
    with Path(path).open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()

def validated_copy(source,target,reason,*,expected_shape=TEST_SHAPE,**metadata):
    source,target=Path(source),Path(target)
    receipt=read(source.with_suffix(".validation.json"))
    if (receipt.get("rows"),receipt.get("candidates"))!=expected_shape or sha256(source)!=receipt["sha256"]:
        raise ValueError(f"Invalid test zip or validation receipt: {source}")
    target.parent.mkdir(parents=True,exist_ok=True)
    if source!=target:
        tmp=target.with_name("."+target.name+".publishing")
        tmp.unlink(missing_ok=True)
        try:os.link(source,tmp)
        except OSError:shutil.copyfile(source,tmp)
        os.replace(tmp,target)
    receipt.update(metadata,source_zip=str(source),zip=str(target),reason=reason,published_utc=time.time())
    atomic_json(target.with_suffix(".validation.json"),receipt)
    return receipt

def registered(trial,records):
    if not MANIFEST.is_file():return None
    entry=read(MANIFEST).get("packages",{}).get(signature(trial,records))
    if entry and Path(entry["zip"]).is_file() and Path(entry["zip"]).with_suffix(".validation.json").is_file():
        return Path(entry["zip"])
