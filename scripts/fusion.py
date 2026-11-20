"""Select a prefix-only ranker and blend on user half A, report on untouched half B."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path
import time
import numpy as np
from numba import njit
import lightgbm as lgb
from exposures import Reader,BASE,PREFIX,MATCH
from ranking import metrics,pack,validate,user_half
from trees import zscore,fit_tree,record,row_metrics

ART=Path("artifacts")
SEEDS={"bge":1119,"qwen":1121,"upper":1122,"upper_s2":1141,"upper_s3":1142,"fullbge":1125,"fullbge_s2":1163,"qlora":1126,"asym4":1144,"e5":1125}
def matched_refit(kind,eval_file):
    match=re.fullmatch(re.escape(kind)+r"_e(\d+)_dev\.npy",Path(eval_file).name)
    if not match:raise ValueError(f"Invalid evaluation checkpoint for {kind}: {eval_file}")
    epoch=int(match.group(1))
    if kind=="e5" and (ART/"round7/e5_fit_incident.json").exists():
        from e5_rescue_policy import selected_test
        return selected_test(eval_file,epoch)
    replay=kind in ("qlora","fullbge")
    suffix="replay" if replay else "receipt"
    for receipt in sorted((ART/"jobs").glob(f"*/{kind}_refit_e{epoch}.{suffix}.json")):
        if kind=="e5" and (ART/"round7/placement.json").is_file() and receipt.parent.name!="mind_r7_e5_fit":continue
        rec=json.loads(receipt.read_text())
        if (rec["series"],rec["epoch"],rec["seed"],rec["training"],rec["positive_pairs"])!=(kind,epoch,SEEDS[kind],["train","dev"],3958501):
            raise ValueError(f"Refit provenance mismatch: {receipt}")
        def verified(item,name):
            identity=rec[item];p=receipt.parent/name
            if identity["name"]!=name or not p.is_file() or p.stat().st_size!=identity["bytes"]:return False
            side=p.with_name(name+".verified")
            if side.exists():return side.read_text().strip()==identity["sha256"]
            return p.stat().st_size<=7*1024**2 and hashlib.sha256(p.read_bytes()).hexdigest()==identity["sha256"]
        stem=f"{kind}_refit_e{epoch}"
        score_name=stem+("_replay" if replay else "")+"_test.npy"
        if (replay or kind in ("fullbge_s2","e5")) and rec.get("checkpoint_scored") is not True:raise ValueError(receipt)
        if verified("checkpoint",stem+".pt") and verified("test_scores",score_name):
            return dict(epoch=epoch,seed=rec["seed"],eval_file=str(eval_file),receipt=str(receipt),
                        test=str(receipt.parent/rec["test_scores"]["name"]),checkpoint_sha256=rec["checkpoint"]["sha256"],
                        test_sha256=rec["test_scores"]["sha256"])
    raise FileNotFoundError(f"No verified train+dev {kind} epoch {epoch} pair")

def branch(kind,split,refit=False):
    if refit:
        if split!="test":raise ValueError("The train+dev fit only produces test logits")
        return Path(matched_refit(kind,branch(kind,"dev"))["test"])
    tag=kind+("_refit" if refit else "")
    for meta in sorted((ART/"jobs").glob("*/"+tag+"_best.json")):
        if kind=="e5" and (ART/"round7/placement.json").is_file() and meta.parent.name!="mind_r7_e5_eval":continue
        ep=json.loads(meta.read_text())["best_epoch"]
        p=meta.parent/f"{tag}_e{ep}_{split}.npy"
        if p.with_name(p.name+".verified").exists():return p
    raise FileNotFoundError(f"No verified {tag} {split} logits")

def feature(r,rows,scores):
    sz=r.off[rows+1]-r.off[rows]
    pos=r.positions(rows)
    z=np.column_stack([zscore(v[pos],sz) for v in scores])
    x=np.column_stack([r.prefix(rows),z])
    return x,z,sz,pos

def sample(rows,count,seed):
    if len(rows)<=count:return rows
    return np.sort(np.random.default_rng(seed).choice(rows,count,replace=False))

def weights(n):
    grid=(tuple(np.bincount(indices,minlength=n)) for indices in itertools.combinations_with_replacement(range(n),4))
    for counts in sorted(grid):
        yield np.asarray(counts,np.float32)/4

@njit(cache=True)
def grid_auc(labels,offset,scores):
    total=0.
    for i in range(len(offset)-1):
        lo,hi=offset[i],offset[i+1]
        wins=positives=0
        for p in range(lo,hi):
            if not labels[p]:continue
            positives+=1
            for n in range(lo,hi):
                if not labels[n] and (scores[p]>scores[n] or (scores[p]==scores[n] and p<n)):wins+=1
        total+=wins/(positives*(hi-lo-positives))
    return total/(len(offset)-1)

def same_day_dev(names,dest):
    dest.mkdir(parents=True,exist_ok=True)
    prog=dest/"progress.json"
    record(prog,dict(stage="opening_dev",models=names))
    r=Reader("dev")
    files=[branch(n,"dev") for n in names]
    scores=[np.load(p,mmap_mode="r") for p in files]
    cols=[*PREFIX,*MATCH,*["z_"+n for n in names]]
    uid=r.uid[r.u]
    hb=user_half(uid);inner=user_half(uid.astype(np.uint64)*131+11939)
    a=np.flatnonzero(~hb);b=np.flatnonzero(hb)
    # User-disjoint fit and blend tuning inside A; outer B remains untouched by both.
    fit=sample(np.flatnonzero((~hb)&(~inner)),60000,1119)
    tune=sample(np.flatnonzero((~hb)&inner),20000,1120)
    def tree(rows):
        x,_,sz,pos=feature(r,rows,scores)
        return fit_tree(x,r.labels[pos],sz,cols,100)
    record(prog,dict(stage="fit_half_A",groups=len(fit)))
    model_a=tree(fit)
    x,z,sz,pos=feature(r,tune,scores)
    preds=np.column_stack([z,zscore(model_a.predict(x,num_threads=4),sz)])
    yy=r.labels[pos]
    best=-1.;selected=None
    for w in weights(preds.shape[1]):
        pr=np.sum(preds*w[None,:],axis=1)
        auc=row_metrics(yy,sz,pr)[:,0].mean()
        if auc>best:best=float(auc);selected=w
    anchor=int(np.argmax([row_metrics(yy,sz,preds[:,i])[:,0].mean() for i in range(len(names))]))
    record(dest/"weights_from_A.json",dict(names=names+["LightGBM"],weights=selected.tolist(),auc=best,
                fit_users_half="A",fit_groups=len(fit),tune_groups=len(tune),anchor=names[anchor]))
    # A-to-B evaluation is complete before training the reverse cross-fit for full-dev reporting.
    del x,z,preds,pos,yy
    tree_out=np.lib.format.open_memmap(dest/"tree_oof_dev.npy",mode="w+",dtype="float32",shape=(len(r.cand),))
    blend=np.lib.format.open_memmap(dest/"blend_oof_dev.npy",mode="w+",dtype="float32",shape=(len(r.cand),))
    def fill(rows,m,phase):
        begin=time.time()
        for st in range(0,len(rows),6000):
            pick=rows[st:st+6000];x,z,sz,pos=feature(r,pick,scores)
            c=zscore(m.predict(x,num_threads=4),sz)
            tree_out[pos]=c;blend[pos]=np.sum(z*selected[:-1],axis=1)+selected[-1]*c
            done=st+len(pick)
            record(prog,dict(stage=phase,done=done,total=len(rows),eta_sec=(len(rows)-done)*(time.time()-begin)/done))
    fill(b,model_a,"independent_half_B")
    # Comparison against the anchor is on the same full candidate slates and the same held-out users.
    def subset(v,rows):
        pos=r.positions(rows);sz=r.off[rows+1]-r.off[rows]
        m=row_metrics(r.labels[pos],sz,v[pos]).mean(0);return dict(zip(("AUC","MRR","nDCG@5","nDCG@10"),m.tolist()))
    check={"anchor":subset(scores[anchor],b),"blend":subset(blend,b),"tree":subset(tree_out,b)}
    accepted=check["blend"]["AUC"]>=check["anchor"]["AUC"]
    record(dest/"heldout_B.json",dict(groups=len(b),metrics=check,accepted=accepted))
    model_b=tree(sample(b,60000,1121));fill(a,model_b,"reverse_crossfit_on_A")
    tree_out.flush();blend.flush()
    reports={name:metrics(BASE,"dev",score) for name,score in zip(names,scores)}
    reports["LightGBM_crossfit"]=metrics(BASE,"dev",tree_out)
    reports["blend_crossfit"]=metrics(BASE,"dev",blend)
    record(dest/"dev_metrics.json",reports)
    if not accepted:
        selected[:]=0.;selected[anchor]=1.
    record(dest/"candidate_policy.json",dict(models=names,weights=selected.tolist(),columns=cols,accepted=accepted,
            anchor=names[anchor],base_train="train-only for all dev logits",type="prefix-only",
            base_files=[str(p) for p in files],fit="final stacker fits all dev using out-of-training base logits",
            split="user hash half A selection, B check; reverse crossfit for full-dev metrics"))
    record(prog,dict(stage="DEV_DONE",heldout=check))

def shadow_path(kind):
    for meta in sorted((ART/"jobs").glob("*/"+kind+"_shadow_meta.json")):
        if kind=="e5" and (ART/"round7/placement.json").is_file() and meta.parent.name!="mind_r7_e5_eval":continue
        marker=json.loads(meta.read_text())
        if marker.get("fit_before")!="2019-11-12":raise ValueError("Unverified fit window "+str(meta))
        if kind not in ("bge","upper","qwen") and not branch(kind,"dev").name.startswith(f"{kind}_e{marker['epoch']}_"):
            continue
        p=meta.parent/marker["score"]
        if p.with_name(p.name+".verified").is_file():return p
    if kind in ("bge","upper","qwen"):
        for p in sorted((ART/"jobs").glob(f"*/{kind}_shadow_e2_train.npy")):
            if p.with_name(p.name+".verified").is_file():return p
    raise FileNotFoundError(kind+" train-week shadow logits")

def temporal_training(names):
    r=Reader("train")
    scores=[np.load(shadow_path(n),mmap_mode="r") for n in names]
    byday=[]
    for day in (12,13,14):
        t=int(np.datetime64(f"2019-11-{day}T00:00:00").astype("int64"))
        byday.append(sample(np.flatnonzero((r.t>=t)&(r.t<t+86400)),40000,1100+day))
    rows=np.sort(np.concatenate(byday))
    x,_,sz,pos=feature(r,rows,scores)
    return x,np.asarray(r.labels[pos]),sz

def dev(names,dest,ablation=False,compact=False):
    dest.mkdir(parents=True,exist_ok=True)
    prog=dest/"progress.json"
    cols=[*PREFIX,*MATCH,*["z_"+n for n in names]]
    record(prog,dict(stage="fitting_tree_from_earlier_dates",models=names))
    x,lab,sz=temporal_training(names)
    model=fit_tree(x,lab,sz,cols,100)
    model.save_model(str(dest/"temporal_tree.lgb"))
    no_time=None
    if ablation:
        no_time=fit_tree(x[:,len(PREFIX):],lab,sz,cols[len(PREFIX):],100)
        no_time.save_model(str(dest/"tree_without_exposure.lgb"))
    del x,lab,sz
    r=Reader("dev");files=[branch(n,"dev") for n in names]
    scores=[np.load(p,mmap_mode="r") for p in files]
    tree_out=np.lib.format.open_memmap(dest/"temporal_tree_dev.npy",mode="w+",dtype="float32",shape=(len(r.cand),))
    no_time_scores=np.empty(len(r.cand),np.float32) if ablation else None
    for st in range(0,len(r.u),6000):
        en=min(st+6000,len(r.u));x,_,sz,_=feature(r,np.arange(st,en),scores)
        tree_out[r.off[st]:r.off[en]]=zscore(model.predict(x,num_threads=4),sz)
        if ablation:
            no_time_scores[r.off[st]:r.off[en]]=zscore(no_time.predict(x[:,len(PREFIX):],num_threads=4),sz)
        record(prog,dict(stage="scoring_official_dev",done=en,total=len(r.u)))
    tree_out.flush()
    hb=user_half(r.uid[r.u]);tune=sample(np.flatnonzero(~hb),30000,1120)
    pos=r.positions(tune);sz=r.off[tune+1]-r.off[tune]
    mat=np.column_stack([*[zscore(v[pos],sz) for v in scores],tree_out[pos]])
    yy=r.labels[pos];best=-1.0;offset=np.r_[0,np.cumsum(sz,dtype=np.int64)]
    for w in weights(mat.shape[1]):
        auc=grid_auc(yy,offset,np.sum(mat*w,axis=1))
        if auc>best:best=float(auc);selected=w
    anchor=int(np.argmax([row_metrics(yy,sz,mat[:,i])[:,0].mean() for i in range(len(names))]))
    record(dest/"weights_from_A.json",dict(names=names+["LightGBM"],weights=selected.tolist(),auc=best,
                fit_days="Nov 12–14",fit_groups=120000,tune_groups=len(tune),anchor=names[anchor]))
    blend=np.lib.format.open_memmap(dest/"blend_dev.npy",mode="w+",dtype="float32",shape=(len(r.cand),))
    for st in range(0,len(r.u),10000):
        en=min(st+10000,len(r.u));sl=slice(r.off[st],r.off[en]);sz=np.diff(r.off[st:en+1])
        blend[sl]=tree_out[sl]*selected[-1]
        for i,v in enumerate(scores):blend[sl]+=zscore(v[sl],sz)*selected[i]
    blend.flush()
    reports={n:metrics(BASE,"dev",v) for n,v in zip(names,scores)}
    reports["LightGBM"]=metrics(BASE,"dev",tree_out);reports["blend"]=metrics(BASE,"dev",blend)
    check={k:reports[key]["user_half_check"] for k,key in (("anchor",names[anchor]),("blend","blend"),("tree","LightGBM"))}
    accepted=check["blend"]["AUC"]>=check["anchor"]["AUC"]
    record(dest/"heldout_B.json",dict(groups=int(hb.sum()),metrics=check,accepted=accepted))
    if not accepted and not compact:
        reports["proposed_blend"]=reports["blend"]
        selected[:]=0.;selected[anchor]=1.
        blend[:]=scores[anchor][:]
        blend.flush()
        reports["blend"]=reports[names[anchor]]
    if ablation:
        reports["identical_tree_without_exposure"]=metrics(BASE,"dev",no_time_scores)
        with_no_time=np.empty_like(no_time_scores)
        for st in range(0,len(r.u),10000):
            en=min(st+10000,len(r.u));sl=slice(r.off[st],r.off[en]);sz=np.diff(r.off[st:en+1])
            with_no_time[sl]=selected[-1]*no_time_scores[sl]
            for i,v in enumerate(scores):
                if selected[i]:with_no_time[sl]+=zscore(v[sl],sz)*selected[i]
        reports["same_blend_weights_without_exposure"]=metrics(BASE,"dev",with_no_time)
        record(dest/"prefix_ablation.json",dict(removed=list(PREFIX),retained=list(MATCH)+["z_"+n for n in names],
            fit_groups=120000,
            tree_delta={s:reports["LightGBM"][s]["AUC"]-reports["identical_tree_without_exposure"][s]["AUC"] for s in reports["LightGBM"]},
            fixed_blend_delta={s:reports["blend"][s]["AUC"]-reports["same_blend_weights_without_exposure"][s]["AUC"] for s in reports["blend"]}))
    record(dest/"dev_metrics.json",reports)
    record(dest/"candidate_policy.json",dict(models=names,weights=selected.tolist(),columns=cols,accepted=accepted,
            anchor=names[anchor],type="prefix-only",base_files=list(map(str,files)),
            tree_protocol="fit on Nov 12–14 with base logits trained through Nov 11; weights on dev half A, check on B",
            refit="out-of-training base scores; 40000 complete groups each on Nov 12–15"))
    record(prog,dict(stage="DEV_DONE",heldout=check))
    if compact:
        del tree_out,blend
        (dest/"temporal_tree_dev.npy").unlink(missing_ok=True)
        (dest/"blend_dev.npy").unlink(missing_ok=True)

def refit_tree(dest):
    pol=json.loads((dest/"policy.json").read_text())
    if pol["weights"][-1]==0:return
    r=Reader("dev"); scores=[np.load(p,mmap_mode="r") for p in pol["base_files"]]
    rows=sample(np.arange(len(r.off)-1),40000,1115)
    x,_,sz,pos=feature(r,rows,scores)
    xt,yt,st=temporal_training(pol["models"])
    fit_tree(np.concatenate([xt,x]),np.concatenate([yt,r.labels[pos]]),np.concatenate([st,sz]),
             pol["columns"],100).save_model(str(dest/"ranker.lgb"))

def score_test(dest,policy_name,tree_name,output,refit_base=False,stage="test_scoring",output_dir=None,pinned_records=None):
    policy=dest/policy_name
    pol=json.loads(policy.read_text());names=pol["models"];w=np.asarray(pol["weights"],np.float32)
    if pol["type"]!="prefix-only":raise ValueError("Unsupported policy")
    r=Reader("test")
    refits={}
    if refit_base:
        refits=pinned_records if pinned_records is not None else {k:matched_refit(k,p) for k,p in zip(names,pol["base_files"],strict=True)}
        if set(refits)!=set(names):raise ValueError("Pinned content sources do not match the policy")
        paths=[Path(refits[k]["test"]) for k in names]
    else:
        paths=[Path(str(p).removesuffix("_dev.npy")+"_test.npy") for p in pol["base_files"]]
        if any(not p.with_name(p.name+".verified").is_file() for p in paths):
            raise FileNotFoundError("A checkpoint pinned by the policy has no verified test logits")
    scores=[np.load(p,mmap_mode="r") for p in paths]
    model=lgb.Booster(model_file=str(dest/tree_name)) if w[-1]>0 else None
    total=len(r.off)-1
    out=np.lib.format.open_memmap(dest/(output+"_test.npy"),mode="w+",dtype="float32",shape=(len(r.cand),))
    begin=time.time()
    for st in range(0,total,8000):
        en=min(st+8000,total);rows=np.arange(st,en)
        if model is not None:
            x,z,sz,_=feature(r,rows,scores)
            c=zscore(model.predict(x,num_threads=4),sz)
        else:
            sz=np.diff(r.off[st:en+1]);sl=slice(r.off[st],r.off[en])
            z=np.column_stack([zscore(v[sl],sz) for v in scores]);c=0.
        sl=slice(r.off[st],r.off[en])
        out[sl]=w[-1]*c
        for i in range(len(names)):
            if w[i]:out[sl]+=w[i]*z[:,i]
        record(dest/(output+".progress.json"),dict(stage=stage,done=en,total=total,eta_sec=(total-en)*(time.time()-begin)/en))
    out.flush()
    zpath=(Path(output_dir) if output_dir is not None else ART/"submissions")/(output+".zip")
    pack(BASE,"test",out,zpath)
    result=validate(zpath,Path("data/large_test/MINDlarge_test/behaviors.tsv"))
    from e5_rescue_policy import content_training,scopes
    result.update(sha256=hashlib.sha256(zpath.read_bytes()).hexdigest(),policy=str(policy),tree=str(dest/tree_name),
                  components=list(map(str,paths)),matched_refit=refits,content_training=content_training(refits) if refit_base else "train",
                  content_training_by_branch=scopes(refits) if refit_base else {n:["train"] for n in names},
                  tree_fit_days="Nov 12–15" if tree_name=="ranker.lgb" else "Nov 12–14",future_exposure=False)
    record(zpath.with_suffix(".validation.json"),result)
    record(dest/(output+".progress.json"),dict(stage="DONE",zip=str(zpath),sha256=result["sha256"]))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("action",choices=["dev","same-day-dev","refit-tree","final"]);p.add_argument("--models",default="bge,upper,qwen")
    p.add_argument("--dest",default="artifacts/fusion");p.add_argument("--ablation",action="store_true")
    p.add_argument("--compact",action="store_true",help="Keep the A-selected blend intact and retain the tree, policy and metrics after scoring")
    a=p.parse_args()
    if a.action=="dev":dev(a.models.split(","),Path(a.dest),a.ablation,a.compact)
    elif a.action=="same-day-dev":same_day_dev(a.models.split(","),Path(a.dest))
    elif a.action=="refit-tree":refit_tree(Path(a.dest))
    else: score_test(Path(a.dest),"policy.json","ranker.lgb","final_prefix",refit_base=True)

