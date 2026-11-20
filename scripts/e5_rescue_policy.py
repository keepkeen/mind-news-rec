"""Accept the selected E5 epoch with signed training provenance and a fixed pre-lock cutoff."""
from datetime import datetime
import json
from pathlib import Path
import time
from zoneinfo import ZoneInfo
import numpy as np

ROOT=Path("artifacts")
WORK=ROOT/"round7"
EVAL=ROOT/"jobs/mind_r7_e5_eval"
FIT=ROOT/"jobs/mind_r7_e5_fit_resume"
ZONE=ZoneInfo("Asia/Shanghai")
TARGET=datetime(2026,11,20,4,15,tzinfo=ZONE).timestamp()
LOCK=TARGET+900
def read(path):
    return json.loads(path.read_text())

def verified(path,identity=None):
    marker=path.with_name(path.name+".verified")
    if not path.is_file() or not marker.is_file():raise FileNotFoundError(path)
    sha=marker.read_text().strip()
    if len(sha)!=64:raise ValueError(f"Invalid SHA receipt for {path.name}")
    if identity is not None and (path.name,path.stat().st_size,sha)!=(identity["name"],identity["bytes"],identity["sha256"]):
        raise ValueError(f"Invalid E5 artifact identity: {path.name}")
    if marker.stat().st_mtime>=LOCK:raise FileNotFoundError(f"Late E5 artifact: {path.name}")
    return sha

def ready_dev(p,shadow):
    if p.parent.resolve()!=EVAL.resolve() or shadow.parent.resolve()!=EVAL.resolve():return False
    try:
        h=read(EVAL/"heartbeat.json")["status"].get("e5.status.json",{})
        best=read(EVAL/"e5_best.json");epoch=best["best_epoch"]
        meta=read(EVAL/"e5_shadow_meta.json")
        if h.get("phase")!="DONE" or h["utc_epoch"]>=LOCK or h.get("best_epoch")!=epoch:return False
        if meta.get("seed")!=1125 or meta.get("epoch")!=epoch or meta.get("fit_before")!="2019-11-12":return False
        if p.name!=f"e5_e{epoch}_dev.npy" or shadow.name!=meta["score"]:return False
        for ep in (1,2,3):verified(EVAL/f"e5_e{ep}.pt")
        curve={ep:read(EVAL/f"e5_e{ep}_metrics.json")["user_half_fit"]["AUC"] for ep in (1,2,3)}
        if max(curve,key=curve.get)!=epoch:raise ValueError("E5 epoch differs from its independent A-half selection")
        for path,split in ((p,"dev"),(shadow,"train")):
            verified(path)
            if np.load(path,mmap_mode="r").shape!=(int(np.load(ROOT/"compact"/split/"offset.npy",mmap_mode="r")[-1]),):
                raise ValueError("E5 out-of-sample score alignment")
        return True
    except FileNotFoundError:return False

def selected_test(eval_file,epoch,now=None):
    p=Path(eval_file);now=time.time() if now is None else now
    if p.parent.resolve()!=EVAL.resolve() or read(EVAL/"e5_best.json")["best_epoch"]!=epoch:raise FileNotFoundError(p)
    dev_hash=verified(p)
    # Order is fixed before seeing any test outcome. Give the refit first choice through lock.
    choices=[(FIT,["train","dev"],f"e5_refit_e{epoch}",f"e5_refit_e{epoch}.pt")]
    if now>=TARGET or (WORK/"e5_refit_unavailable.json").is_file():
        choices.append((EVAL,["train"],f"e5_trainonly_e{epoch}",f"e5_e{epoch}.pt"))
    for d,training,stem,cp_name in choices:
        receipt=d/(stem+".receipt.json")
        if not receipt.is_file() or receipt.stat().st_mtime>=LOCK:continue
        meta=read(receipt);test=d/(stem+"_test.npy");cp=d/cp_name
        if (meta["series"],meta["seed"],meta["epoch"],meta["training"],meta.get("checkpoint_scored"))!=("e5",1125,epoch,training,True):
            raise ValueError("E5 source/seed/epoch mismatch")
        if training==["train","dev"] and meta.get("positive_pairs")!=3958501:raise ValueError("Unexpected E5 refit data")
        if training==["train"] and (meta.get("selected_dev_sha256")!=dev_hash or not meta.get("dev_ranks_match")):
            raise ValueError("Train-only test replay did not reproduce the selected dev epoch")
        try:
            ch=verified(cp,meta["checkpoint"]);ts=verified(test,meta["test_scores"])
        except FileNotFoundError:continue
        if np.load(test,mmap_mode="r").shape!=(93115001,):raise ValueError("E5 test candidate count")
        return dict(epoch=epoch,seed=1125,eval_file=str(eval_file),receipt=str(receipt),test=str(test),
                    checkpoint_sha256=ch,test_sha256=ts,training=training)
    raise FileNotFoundError(f"No on-time verified E5 test scores at epoch {epoch}")

def scopes(records):
    return {k:v.get("training",["train","dev"]) for k,v in records.items()}

def content_training(records):
    return "mixed" if any(v==["train"] for v in scopes(records).values()) else "train+dev"
