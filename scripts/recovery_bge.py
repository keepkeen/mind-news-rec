"""Recreate the early BGE shadow and score the original epoch-two train+dev checkpoint."""
import hashlib
import json
import sys
import numpy as np
import torch
from click_text import ArticleEncoder,SPECS
from recommender import Model,Split,gpu,score
from jobio import CACHE,DATA,EXPORT,RUN,export_file,refit_receipt,run_job,status,wait_for,write_json
from fleet_queue import child

def replay():
    kind="fullbge_s2";tag=kind+"_refit_e2";job="bge_recovery_replay"
    expected=json.loads((RUN/"original_refit_receipt.json").read_text())
    cp=CACHE/(tag+".pt");wait_for(cp.with_suffix(".sha256"))
    with cp.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    if sha!=expected["checkpoint"]["sha256"] or cp.stat().st_size!=expected["checkpoint"]["bytes"]:raise ValueError("BGE refit checksum")
    torch.manual_seed(1163);torch.set_float32_matmul_precision("high")
    ck=torch.load(cp,map_location="cpu",weights_only=True);spec=ck["spec"]
    if (ck["seed"],ck["epoch"],tuple(ck["hf"]),spec["training_splits"])!=(1163,2,SPECS[kind],["train","dev"]):raise ValueError("BGE refit provenance")
    enc=ArticleEncoder(kind,spec["max_tokens"]);buf=torch.zeros(len(enc.lengths),enc.width,device="cuda",dtype=torch.float16)
    head=Model(buf,gpu(np.load(DATA/"cat.npy").astype(np.int32)),gpu(np.load(DATA/"subcat.npy").astype(np.int32)),4).cuda()
    with torch.no_grad():
        params={n:p for n,p in enc.named_parameters() if p.requires_grad}
        if params.keys()!=ck["article_trainable"].keys():raise ValueError("Missing trainable block")
        for n,p in params.items():p.copy_(ck["article_trainable"][n])
        head.load_state_dict(ck["head"],strict=True)
        status(job,"scoring_epoch_two_test",checkpoint_sha256=sha)
        enc.all_articles(buf,spec["encoder_micro"]);path,_=score(head,Split("test"),tag)
    export_file(path);refit_receipt(kind,2,1163,cp,path,3958501,8192,checkpoint_scored=True)
    digest=json.loads((EXPORT/(path.name+".parts.json")).read_text())["sha256"]
    write_json(EXPORT/"bge1163_recovery.json",dict(checkpoint_sha256=sha,new_test_sha256=digest,
        original_test_sha256=expected["test_scores"]["sha256"],same_bytes=digest==expected["test_scores"]["sha256"]))
    status(job,"DONE",epoch=2)
def main():
    q="bge_recovery_queue"
    status(q,"waiting_for_inputs")
    for name in ("shared","train","dev"):wait_for(DATA/(name+".ready"))
    # Only the epoch/seed decision crosses the training boundary. The shadow opens generic BGE weights.
    write_json(EXPORT/"fullbge_s2_best.json",dict(job="fullbge_s2",best_epoch=2,seed=1163,source="locally verified train-only dev epoch two"))
    status(q,"running",child="fullbge_s2_shadow")
    child("fullbge_s2_shadow",["click_train.py","fullbge_s2","8192","192","64","2","1163","--shadow","--checkpoint-every","100"])
    status(q,"running",child="bge_recovery_replay");child("bge_recovery_replay",["recovery_bge.py","replay"])
    status(q,"DONE")
if __name__=="__main__":
    if len(sys.argv)>1:run_job("bge_recovery_replay",replay)
    else:run_job("bge_recovery_queue",main)
