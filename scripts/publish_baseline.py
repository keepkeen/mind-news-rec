"""Promote a synced backup only after full raw test verification. Never submit externally."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from ranking import validate

root=Path(__file__).resolve().parents[1];source=root/"artifacts/jobs/mind_h100";target=root/"artifacts/submissions"
target.mkdir(parents=True,exist_ok=True)
ref=root/"data/large_test/MINDlarge_test/behaviors.tsv"
def main():
    deadline=time.time()+5*3600
    current=0
    while time.time()<deadline:
        for epoch in (1,2):
            if epoch<=current:continue
            z=source/f"bge_e{epoch}.zip"
            m=source/f"bge_e{epoch}_metrics.json"
            if not (z.with_name(z.name+".verified").exists() and m.exists()): continue
            report=validate(z,ref)
            digest=hashlib.sha256(z.read_bytes()).hexdigest()
            report.update(sha256=digest,epoch=epoch,dev=json.loads(m.read_text()),source_model="BAAI/bge-base-en-v1.5",fit="large_train")
            tmp=target/"baseline_train.zip.building"
            shutil.copyfile(z,tmp);tmp.replace(target/"baseline_train.zip")
            path=target/"baseline_train.validation.json";j=path.with_suffix(".tmp")
            j.write_text(json.dumps(report,indent=2));j.replace(path)
            current=epoch;print(json.dumps({k:report[k] for k in ("rows","candidates","sha256","epoch")}),flush=True)
        p=source/"bge_best.json";hb=source/"heartbeat.json"
        if current and p.exists() and hb.exists():
            status=json.loads(hb.read_text())["status"].get("bge.status.json",{})
            if status.get("phase")=="DONE" and current==status["best_epoch"]:return
        time.sleep(12)
    raise TimeoutError("Baseline zip did not reach disk by the deadline")
if __name__=="__main__":main()

