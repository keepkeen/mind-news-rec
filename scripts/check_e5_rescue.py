"""Exercise pre-deadline source precedence and fallback provenance with isolated files."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
import numpy as np
import e5_rescue_policy as policy
from colab_jobs import ART,atomic_json

def signed(p):
    with p.open("rb") as f:h=hashlib.file_digest(f,"sha256").hexdigest()
    p.with_name(p.name+".verified").write_text(h)
    return dict(name=p.name,bytes=p.stat().st_size,sha256=h)

def exercise(root):
    policy.WORK=root;policy.EVAL=root/"eval";policy.FIT=root/"fit"
    for d in (policy.EVAL,policy.FIT):d.mkdir()
    ep=2;p=policy.EVAL/"e5_e2_dev.npy";p.write_bytes(b"synthetic dev");dev=signed(p)
    (policy.EVAL/"e5_best.json").write_text(json.dumps(dict(best_epoch=ep)))
    # Sparse arrays exercise the real MIND shape check without materializing the fixture.
    for d,stem,tr,base in [(policy.EVAL,"e5_trainonly_e2",["train"],"e5_e2.pt"),
                           (policy.FIT,"e5_refit_e2",["train","dev"],"e5_refit_e2.pt")]:
        cp=d/base;cp.write_bytes(stem.encode())
        ar=d/(stem+"_test.npy");x=np.lib.format.open_memmap(ar,mode="w+",dtype="float32",shape=(93115001,));del x
        rec=dict(series="e5",epoch=2,seed=1125,training=tr,checkpoint_scored=True,positive_pairs=3958501,
            checkpoint=signed(cp),test_scores=signed(ar),selected_dev_sha256=dev["sha256"],dev_ranks_match=True)
        (d/(stem+".receipt.json")).write_text(json.dumps(rec))
    for stamp in (*root.glob("*/*.verified"),*root.glob("*/*.receipt.json")):
        os.utime(stamp,(policy.TARGET-60,policy.TARGET-60))
    before=policy.TARGET-1;after=policy.TARGET+1
    assert policy.selected_test(p,2,before)["training"]==["train","dev"]
    assert policy.selected_test(p,2,after)["training"]==["train","dev"]
    mark=policy.FIT/"e5_refit_e2_test.npy.verified";os.utime(mark,(policy.LOCK+1,policy.LOCK+1))
    try:policy.selected_test(p,2,before)
    except FileNotFoundError:pass
    else:raise ValueError("Fallback became eligible before its decision time")
    fallback=policy.selected_test(p,2,after)
    assert fallback["training"]==["train"] and policy.content_training({"e5":fallback,"bge":{}})=="mixed"
    try:policy.selected_test(p,1,after)
    except FileNotFoundError:pass
    else:raise ValueError("Wrong epoch was accepted")
    receipt=policy.EVAL/"e5_trainonly_e2.receipt.json";rec=json.loads(receipt.read_text());rec["selected_dev_sha256"]="bad"
    receipt.write_text(json.dumps(rec))
    os.utime(receipt,(before,before))
    try:policy.selected_test(p,2,after)
    except ValueError:pass
    else:raise ValueError("Wrong dev lineage was accepted")
    return dict(refit_preferred_before_and_after_0415=True,early_fallback_excluded=True,
                late_refit_excluded=True,train_only_enabled_after_0415=True,mixed_scope_disclosed=True,
                wrong_epoch_rejected=True,wrong_dev_source_rejected=True)

if __name__=="__main__":
    with tempfile.TemporaryDirectory(dir=ART/"local",prefix="e5_rescue_check_") as tmp:out=exercise(Path(tmp))
    atomic_json(ART/"round7/rescue_policy_selfcheck.json",out);print(json.dumps(out))
