"""Build the three-model candidate, independently check the official dev scorer, and validate the zip."""
import concurrent.futures as cf
from pathlib import Path
import json
import subprocess
import sys
import time
from colab_jobs import ROOT,ART
from fusion import branch,shadow_path,dev,score_test
from ranking import pack
from trees import record
import numpy as np

NAMES=["bge","upper","qwen"]
DEST=ART/"fusion"
OUT="fusion_three_train_prefix"

def ready(n):
    try: branch(n,"dev");branch(n,"test");shadow_path(n)
    except FileNotFoundError:return False
    # The selected epoch must have been finalized before the test package pins its inputs.
    p=branch(n,"dev").parent/"heartbeat.json"
    return p.exists() and json.loads(p.read_text())["status"].get(n+".status.json",{}).get("phase")=="DONE"

def official_check():
    z=DEST/"official_dev.zip"
    pack(ART/"compact","dev",np.load(DEST/"blend_dev.npy",mmap_mode="r"),z)
    subprocess.run([sys.executable,"scripts/ranking.py","official","--split","dev","--zip",str(z),
                    "--expected",str(DEST/"dev_metrics.json"),"--expected-key","blend","--out",str(DEST/"official_dev.json")],
                   cwd=ROOT,check=True)

def main():
    DEST.mkdir(parents=True,exist_ok=True)
    progress=DEST/"today_progress.json"
    verified=ART/"submissions"/(OUT+".validation.json")
    if verified.exists() and (DEST/"official_dev.json").exists() and (ART/"submissions"/(OUT+".zip")).exists():
        import hashlib
        receipt=json.loads(verified.read_text())
        if hashlib.sha256((ART/"submissions"/(OUT+".zip")).read_bytes()).hexdigest()==receipt["sha256"]:
            record(progress,dict(stage="DONE",zip=str(ART/"submissions"/(OUT+".zip")),sha256=receipt["sha256"],
                                 official_diff=json.loads((DEST/"official_dev.json").read_text())["max_abs_diff"]))
            return
    while True:
        ready_list=[k for k in NAMES if ready(k)]
        record(progress,dict(stage="waiting_for_three_OOF_models",ready=ready_list))
        if ready_list==NAMES:break
        time.sleep(18)
    if not (DEST/"prefix_ablation.json").exists() or not (DEST/"candidate_policy.json").exists():
        record(progress,dict(stage="three_model_temporal_validation"))
        dev(NAMES,DEST,ablation=True)
    record(progress,dict(stage="dev_official_and_full_test",zip=OUT))
    with cf.ThreadPoolExecutor(max_workers=2) as pool:
        a=pool.submit(official_check)
        b=pool.submit(score_test,DEST,"candidate_policy.json","temporal_tree.lgb",OUT)
        a.result();b.result()
    j=json.loads((ART/"submissions"/(OUT+".validation.json")).read_text())
    record(progress,dict(stage="DONE",zip=str(ART/"submissions"/(OUT+".zip")),sha256=j["sha256"],
                         official_diff=json.loads((DEST/"official_dev.json").read_text())["max_abs_diff"]))
if __name__=="__main__":
    try:main()
    except Exception as e:
        record(DEST/"today_progress.json",dict(stage="FAILED",error=repr(e)));raise

