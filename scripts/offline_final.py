"""Rebuild the immutable R7 fallback from its local A/B lock, then audit the train-only control."""
import argparse
import json
from pathlib import Path
import re
import time
import numpy as np
from colab_jobs import ART,JOBS,atomic_json
from coordinate import eligible_trials
from fusion import matched_refit,refit_tree,score_test,shadow_path
from incremental_policy import read,registered,sha256,signature,validated_copy

def main(rebuild):
    work=ART/"round7";lock=work/"offline_lock.json";target=ART/"submissions/final_prefix.zip"
    if not lock.is_file():
        pool=eligible_trials();t=pool[0];source=registered(t,t["matched_refit"])
        atomic_json(lock,dict(time=time.time(),selected=t,candidates=pool,source_zip=str(source) if source else None))
    frozen=read(lock);t=frozen["selected"];d=Path(t["dest"]);pol=read(d/"candidate_policy.json");proof={}
    control={};missing=[];base=ART/"compact"
    for name,devfile in zip(t["models"],pol["base_files"],strict=True):
        p=Path(devfile);r=matched_refit(name,p);sh=shadow_path(name);ref_path=Path(r["receipt"]);receipt=read(ref_path)
        epoch=r["epoch"];best=read(p.parent/(name+"_best.json"))
        curve={}
        for metric in p.parent.glob(name+"_e*_metrics.json"):
            m=re.fullmatch(re.escape(name)+r"_e(\d+)_metrics\.json",metric.name)
            if m:curve[int(m.group(1))]=read(metric)["user_half_fit"]["AUC"]
        if best["best_epoch"]!=epoch or max(sorted(curve),key=curve.get)!=epoch or receipt["seed"]!=r["seed"]:raise ValueError(f"Wrong epoch or seed for {name}")
        files={"dev":p,"shadow":sh,"test":Path(r["test"]),"checkpoint":ref_path.parent/receipt["checkpoint"]["name"]}
        hashes={k:sha256(v) for k,v in files.items()}
        if hashes["test"]!=r["test_sha256"] or hashes["checkpoint"]!=r["checkpoint_sha256"]:raise ValueError(name)
        for key,sp in (("shadow","train"),("dev","dev"),("test","test")):
            f=files[key]
            if f.with_name(f.name+".verified").read_text().strip()!=hashes[key] or np.load(f,mmap_mode="r").shape!=(int(np.load(base/sp/"offset.npy",mmap_mode="r")[-1]),):raise ValueError(f)
        proof[name]=dict(epoch=epoch,seed=r["seed"],curve_A=curve,hashes=hashes,files={k:str(v) for k,v in files.items()},refit=r)
        tag=name+f"_e{epoch}_test.npy";available=[f for f in JOBS.glob("*/"+tag) if f.with_name(f.name+".verified").is_file()]
        control[name]=[str(f) for f in available]
        if not available:missing.append(name)
    metrics=read(d/"dev_metrics.json")["blend"]
    control_record=dict(score_files=control,missing=missing,dev_validation_shared=True,metrics={k:metrics[k] for k in ("all","user_half_fit","user_half_check")})
    if not missing and not (ART/"submissions/final_train_only_prefix.validation.json").exists():
        tmp=work/"rebuild";tmp.mkdir(exist_ok=True);atomic_json(tmp/"policy.json",pol);refit_tree(tmp)
        score_test(tmp,"policy.json","ranker.lgb","final_train_only_prefix",refit_base=False)
    atomic_json(work/"train_only_control.json",control_record)
    refs={n:v["refit"] for n,v in proof.items()};source=registered(t,refs)
    if source is not None:
        ret=read(source.with_suffix(".validation.json"))
        if ret["matched_refit"]!=refs or ret["content_training"]!="train+dev" or ret["future_exposure"] or ret["tree_fit_days"]!="Nov 12–15":raise ValueError("Archive protocol mismatch")
        if sha256(Path(ret["tree"]))!=ret["ranker_sha256"]:raise ValueError("Fitted tree hash mismatch")
        validated_copy(source,target,"R7 local fallback; epoch and score digests rechecked",round7_backup=True)
    elif not rebuild:raise FileNotFoundError("No matching published archive")
    audit=dict(models=t["models"],weights=pol["weights"],metrics=control_record["metrics"],branches=proof,
               tree_rows_per_day=40000,tree_days=[12,13,14,15],source=str(source),input_key=signature(t,refs))
    if rebuild:
        dest=work/"rebuild";dest.mkdir(exist_ok=True);atomic_json(dest/"policy.json",pol);refit_tree(dest)
        tree=sha256(dest/"ranker.lgb");audit["rebuilt_tree_sha256"]=tree
        if source and tree!=ret["ranker_sha256"]:raise ValueError("Rebuilt tree differs from archived tree")
        score_test(dest,"policy.json","ranker.lgb","final_prefix",refit_base=True,output_dir=dest)
        new=dest/"final_prefix.zip"
        if source and sha256(new)!=sha256(source):raise ValueError("Rebuild differs from A-selected archive")
        validated_copy(new,target,"R7 fresh local rebuild, checked against matched input hashes and original TSV",round7_backup=True)
        (dest/"final_prefix_test.npy").unlink(missing_ok=True)
        audit["rebuild_identical_to_published"]=source is not None
    audit["final_sha256"]=sha256(target);atomic_json(work/"final_audit.json",audit)
    print(target,audit["final_sha256"],flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--rebuild",action="store_true");main(p.parse_args().rebuild)
