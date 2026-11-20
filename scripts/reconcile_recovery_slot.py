"""After definitive capacity refusals, use a fresh notebook only if both shapes of the old one are absent."""
import concurrent.futures as cf
import fcntl
import json
import sys
import uuid
import time
from colab_cli.client import Assignment
from colab_cli.common import state
from colab_cli.commands.session import resolve_runtime_options
from colab_jobs import JOBS,atomic_json
from colab_alerts import permitted,check
from recovery_watchdog import SESSIONS

def change(s):
    d=JOBS/s
    with (d/"allocation.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (d/"machine.json").exists():return
        p=d/"allocation.json";r=json.loads(p.read_text())
        if r.get("new_identity_after_prune"):return
        permitted()
        for high in (True,False):
            try:v=state.client._get_assignment(uuid.UUID(r["notebook"]),*resolve_runtime_options("A100",None,high_mem=high))
            except Exception as exc:check(str(exc),s,"reconcile");raise
            if isinstance(v,Assignment):print(s,"old assignment exists; retain its identity",flush=True);return
        r["new_identity_after_prune"]=r["notebook"];r["notebook"]=str(uuid.uuid4());r["first_reconciled_request"]=time.time()
        atomic_json(p,r);print(s,"new identity after confirmed empty old slot",flush=True)
if __name__=="__main__":
    with cf.ThreadPoolExecutor(max_workers=3) as p:list(p.map(change,SESSIONS))
