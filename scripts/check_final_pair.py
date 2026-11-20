"""Run the actual official-decision listener, both ZIP validators and paired receipt on real slates."""
import copy
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile
from colab_jobs import ROOT,atomic_json
from final_pair import PAIR,RAW,build_pair,digest,layout,pinned_inputs,read,test_training
from independent_submission import verify
from recovery_final import best_published,publish_official

def prepare_official_sample(floor,raw,staged,n,n_dev,daily):
    import lightgbm as lgb
    import numpy as np
    from experimental_forward import blend,dev_report,training_rows
    from exposures import BASE,Reader,unpacked_index
    from fusion import feature,sample
    from ranking import arr,rank
    from trees import fit_tree,zscore
    source=Path(floor["zip"]);trial=copy.deepcopy(floor["selection"])
    meta=read(source.with_suffix(".validation.json"));policy=read(meta["policy"])
    d=staged.parent/"sample_prefix_trial";d.mkdir()
    weights=np.asarray(trial["weights"],np.float32)
    def inputs(k):
        return [np.load(trial["matched_refit"][name]["test"] if k=="test" else trial["oof_inputs"][name][k]["path"],mmap_mode="r")
                for name in trial["models"]]
    train=Reader("train");dev=Reader("dev");test=Reader("test")
    tr=training_rows(train,daily);validation=np.arange(n_dev);dv=sample(validation,daily,1115)
    xt,_,gt,pt=feature(train,tr,inputs("shadow"));yt=train.labels[pt]
    model=fit_tree(xt,yt,gt,policy["columns"],100);model.save_model(str(d/"temporal_tree.lgb"))
    x,z,g,_=feature(dev,validation,inputs("dev"))
    report=dev_report(dev,blend(z,zscore(model.predict(x,num_threads=4),g),weights),n_dev)
    xd,_,gd,pd=feature(dev,dv,inputs("dev"))
    final=fit_tree(np.concatenate((xt,xd)),np.concatenate((yt,dev.labels[pd])),np.concatenate((gt,gd)),policy["columns"],100)
    final.save_model(str(d/"ranker.lgb"))
    atomic_json(d/"candidate_policy.json",policy);atomic_json(d/"dev_metrics.json",dict(blend=report))
    atomic_json(d/"training.json",dict(fit_days=[12,13,14],final_tree_days=[12,13,14,15],groups_per_day=daily,
        fit_groups=len(tr),final_fit_groups=len(tr)+len(dv),train_rows_sha256=hashlib.sha256(tr.tobytes()).hexdigest(),
        dev_refit_rows_sha256=hashlib.sha256(dv.tobytes()).hexdigest()))
    trial.update(dest=str(d),full_dev=report["all"]["AUC"],selection_A=report["user_half_fit"]["AUC"],
                 heldout_B=report["user_half_check"]["AUC"])
    count=0
    with RAW.open("rb") as src,raw.open("wb") as dst:
        for row in itertools.islice(src,n):
            dst.write(row)
            count+=len(row.rpartition(b"\t")[2].split())
    x,z,g,_=feature(test,np.arange(n),inputs("test"))
    fitted=lgb.Booster(model_file=meta["tree"])
    published=blend(z,zscore(fitted.predict(x,num_threads=4),g),weights)
    sample_scores=blend(z,zscore(final.predict(x,num_threads=4),g),weights)
    ids=arr(BASE,"test","imp_id")
    def line(i,scores):
        values=rank(scores[test.off[i]:test.off[i+1]])
        return (str(ids[i])+" ["+",".join(map(str,values.tolist()))+"]\n").encode("ascii")
    with zipfile.ZipFile(source) as full,full.open("prediction.txt") as ranks:
        with zipfile.ZipFile(staged,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=2) as small:
            with small.open("prediction.txt","w",force_zip64=True) as out:
                for i in range(n):
                    if ranks.readline()!=line(i,published):raise ValueError(f"Full incumbent replay differs on test row {i}")
                    out.write(line(i,sample_scores))
    meta.update(zip=str(staged),sha256=digest(staged),rows=n,candidates=count,verified_against=str(raw),rehearsal=True,
                policy=str(d/"candidate_policy.json"),tree=str(d/"ranker.lgb"),ranker_sha256=digest(d/"ranker.lgb"))
    atomic_json(staged.with_suffix(".validation.json"),meta)
    unpacked_index.cache_clear()
    return (n,count),trial

def check_mixed_provenance():
    # E5's pending test run is unnecessary for checking the new receipt's two permitted source modes.
    record=dict(seed=1125,epoch=2,training=["train"])
    proof=dict(series="e5",seed=1125,epoch=2,training=["train"],dev_ranks_match=True,selected_dev_sha256="sample-hash")
    assert test_training("e5",record,proof,["train"],"sample-hash")==["train"]
    try:test_training("e5",record,proof,["train","dev"],"sample-hash")
    except ValueError:return
    raise AssertionError("A train-only E5 receipt was mislabeled as train+dev")

def main():
    start=time.time();floor=read(PAIR/"incumbent.json")
    assert best_published([],floor)==floor
    prefix=Path(floor["zip"])
    if digest(prefix)!=floor["sha256"]:raise ValueError("Rehearsal incumbent checksum")
    base=PAIR/"rehearsals";base.mkdir(exist_ok=True,parents=True)
    d=Path(tempfile.mkdtemp(prefix="sample_",dir=base)).resolve()
    staged=d/"sample_prefix.zip"
    shape,trial=prepare_official_sample(floor,d/"behaviors.tsv",staged,2048,4096,256)
    atomic_json(d/"scope.json",dict(test_shape=list(shape),dev_count=4096,daily_groups=256,deadline=start+900))
    job=layout(d)
    with (d/"watch.log").open("w") as log:
        proc=subprocess.Popen([sys.executable,"-u","scripts/final_pair.py","watch","--rehearsal-dir",str(d)],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        try:
            until=time.monotonic()+15
            while not (d/"status.json").is_file() and time.monotonic()<until:time.sleep(.1)
            assert read(d/"status.json")["stage"]=="waiting_for_official"
            assert not job.receipt.exists() and not job.archive.exists()
            decision=publish_official(trial,staged,job.submissions/"final_v2_prefix.zip",job.decision,job.raw,shape,
                                      reason="rehearsal_verified_incumbent")
            code=proc.wait(timeout=max(1,job.deadline-time.time()))
            if code:raise RuntimeError(f"Rehearsal failed. See {d}/build.log and {d}/watch.log")
        finally:
            if proc.poll() is None:proc.terminate();proc.wait(timeout=10)
    receipt=read(job.receipt)
    assert receipt["rehearsal"] and receipt["status"]=="READY"
    assert receipt["models"]==trial["models"] and receipt["weights"]==trial["weights"]
    assert receipt["content_training_by_branch"]==decision["content_training_by_branch"]
    assert receipt["training"]["fit_groups"]==3*256 and receipt["training"]["final_fit_groups"]==4*256
    assert receipt["dev_auc"]["all"]["impressions"]==4096
    assert sum(receipt["dev_auc"][s]["impressions"] for s in ("user_half_fit","user_half_check"))==4096
    assert len(receipt["experimental"]["added_features"])==14
    for key in ("official","experimental"):
        proof=verify(Path(receipt[key]["zip"]),job.raw,expected_shape=shape)
        assert proof["sha256"]==receipt[key]["sha256"]
    wrong=copy.deepcopy(decision);wrong["weights"][0]+=0.25
    try:pinned_inputs(wrong,job)
    except ValueError:pass
    else:raise AssertionError("Different forward weights were accepted")
    check_mixed_provenance()
    # Re-delivery of the same official decision is idempotent and preserves the published receipt.
    prior=digest(job.receipt);reused=build_pair(job)
    assert reused==receipt and digest(job.receipt)==prior
    result=dict(passed=True,utc=time.time(),seconds=time.time()-start,root=str(d),receipt=str(job.receipt),
        source_prefix=str(prefix),source_sha256=floor["sha256"],models=receipt["models"],weights=receipt["weights"],
        sample_train_per_day=256,sample_dev=4096,sample_test=shape[0],sample_test_candidates=shape[1],
        independent_checks=["official","experimental"],official_published_before_forward=True,
        both_sample_trees_fit_identical_groups=True,full_incumbent_test_ranks_reproduced=2048,
        same_branches_weights_and_test_inputs=True,weight_mismatch_rejected=True,
        train_only_provenance_fixture_checked=True,restart_reused_completed_pair=True,
        rehearsal_auc=receipt["dev_auc"])
    atomic_json(PAIR/"dry_run.json",result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=="__main__":main()
