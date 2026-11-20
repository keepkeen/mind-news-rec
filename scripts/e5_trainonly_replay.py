"""Prepare the train-only E5 test alternative at the epoch selected on dev user half A."""
import hashlib
import json
import time
from pathlib import Path
from jobio import CACHE,DATA,EXPORT,RUN,export_file,run_job,status,wait_for,write_json

JOB="e5_trainonly_replay"
def identity(path):
    with path.open("rb") as f:
        sha=hashlib.file_digest(f,"sha256").hexdigest()
    return dict(name=path.name,bytes=path.stat().st_size,sha256=sha)

def main():
    group=RUN/"e5_evaluation_queue.status.json"
    status(JOB,"waiting_for_selected_epoch_and_shadow")
    while True:
        phase=json.loads(group.read_text())["phase"] if group.exists() else "pending"
        if phase=="DONE":break
        if phase in ("FAILED","SKIPPED"):raise RuntimeError("Evaluation queue ended: "+phase)
        time.sleep(12)
    wait_for(DATA/"test.ready")
    import numpy as np
    import torch
    from click_text import ArticleEncoder,SPECS
    from recommender import Model,Split,gpu,score
    torch.manual_seed(1125);torch.set_float32_matmul_precision("high")
    best=json.loads((EXPORT/"e5_best.json").read_text())
    epoch=int(best["best_epoch"]);tag=f"e5_e{epoch}";cp=CACHE/(tag+".pt")
    ci=identity(cp);cm=json.loads((EXPORT/(cp.name+".parts.json")).read_text())
    if cm["sha256"]!=ci["sha256"]:raise ValueError("Evaluation checkpoint digest mismatch")
    ck=torch.load(cp,map_location="cpu",weights_only=True);spec=ck["spec"]
    if (ck["seed"],ck["epoch"],tuple(ck["hf"]),spec["training_splits"])!=(1125,epoch,SPECS["e5"],["train"]):raise ValueError("E5 train-only provenance")
    status(JOB,"scoring_matched_checkpoint",epoch=epoch,checkpoint_sha256=ci["sha256"])
    enc=ArticleEncoder("e5",spec["max_tokens"])
    buf=torch.zeros(len(enc.lengths),enc.width,device="cuda",dtype=torch.float16)
    head=Model(buf,gpu(np.load(DATA/"cat.npy").astype(np.int32)),gpu(np.load(DATA/"subcat.npy").astype(np.int32)),4).cuda()
    with torch.no_grad():
        params={n:p for n,p in enc.named_parameters() if p.requires_grad}
        if params.keys()!=ck["article_trainable"].keys():raise ValueError("Encoder checkpoint is incomplete")
        for n,p in params.items():p.copy_(ck["article_trainable"][n])
        head.load_state_dict(ck["head"],strict=True);del ck
        enc.all_articles(buf,spec["encoder_micro"])
        label=f"e5_trainonly_e{epoch}"
        dev=Split("dev")
        path,_=score(head,dev,label)
        replay=identity(path)
        original=json.loads((EXPORT/(tag+"_dev.npy.parts.json")).read_text())["sha256"]
        exact=original==replay["sha256"]
        if not exact:
            # Both vectors are scores of the same checkpoint. Check actual ranked slates on a byte mismatch.
            old=np.load(CACHE/(tag+"_dev.npy"),mmap_mode="r");new=np.load(path,mmap_mode="r")
            for lo,hi in zip(dev.off[:-1],dev.off[1:]):
                if not np.array_equal(np.argsort(-old[lo:hi],kind="stable"),np.argsort(-new[lo:hi],kind="stable")):
                    raise ValueError("Train-only checkpoint did not reproduce selected dev ranks")
        path.unlink()
        output,_=score(head,Split("test"),label)
    export_file(output)
    test=identity(output)
    if test["sha256"]!=json.loads((EXPORT/(output.name+".parts.json")).read_text())["sha256"]:raise ValueError("Test score export changed")
    write_json(EXPORT/(label+".receipt.json"),dict(series="e5",seed=1125,epoch=epoch,training=["train"],
        checkpoint=ci,test_scores=test,checkpoint_scored=True,selected_dev_sha256=original,
        reproduced_dev_sha256=replay["sha256"],dev_ranks_match=True,exact_dev_bytes=exact,utc=time.time()))
    status(JOB,"DONE",epoch=epoch,test_sha256=test["sha256"],exact_dev_bytes=exact)

if __name__=="__main__":run_job(JOB,main)
