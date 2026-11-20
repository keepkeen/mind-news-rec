"""Reopen the released project notebooks after reconciling refused fresh-notebook requests."""
import concurrent.futures as cf
import fcntl
import json
import subprocess
import sys
import time
import uuid
from colab_cli.client import Assignment
from colab_cli.common import state
from colab_cli.commands.session import resolve_runtime_options
from colab_jobs import JOBS,ROOT,atomic_json

SLOTS={"mind_e5_eval_1119":"mind_seed_eval_1119","mind_e5_fit_1119":"mind_seed_fit_1119"}

def retry(s,prior):
    folder=JOBS/s
    with (folder/"allocation.lock").open("a") as f:
        fcntl.flock(f,fcntl.LOCK_EX)
        if (folder/"machine.json").exists():return
        if not (JOBS/prior/"released.json").exists() or json.loads((folder/"boot.json").read_text())["stage"]!="CAPACITY_UNAVAILABLE":
            raise RuntimeError("Slot not released or first request still pending")
        ap=folder/"allocation.json";rec=json.loads(ap.read_text())
        cli_python=str(ROOT/".venv/bin/python")
        try_id=uuid.UUID(rec["notebook"])
        for gpu,high in (("A100",True),("A100",False),("H100",False)):
            variant,accel,shape=resolve_runtime_options(gpu,None,high_mem=high)
            result=state.client._get_assignment(try_id,variant,accel,shape)
            if isinstance(result,Assignment):
                print(s,"assignment appeared, reconciling established notebook",flush=True)
                options=(gpu,high);break
        else:
            options=("A100",True)
            rec["refused_notebook"]=rec["notebook"]
            rec["notebook"]=json.loads((JOBS/prior/"allocation.json").read_text())["notebook"]
            rec["first_reconciled_request"]=time.time()
            rec["reused_released_notebook"]=prior;atomic_json(ap,rec)
    subprocess.run([cli_python,"scripts/detach.py","r6_reuse_"+s,"/usr/bin/caffeinate","-i",cli_python,
                    "scripts/fleet_boot.py",s,options[0],"2",*(["high"] if options[1] else [])],cwd=ROOT,check=True)
if __name__=="__main__":
    with cf.ThreadPoolExecutor(max_workers=2) as p:list(p.map(lambda item:retry(*item),SLOTS.items()))
