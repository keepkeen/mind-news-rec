"""Reattach a missing project alias using its exact creation endpoint, GPU UUID and MIND queue."""
import json
from pathlib import Path
import sys
import time
import uuid
from colab_cli.client import Assignment
from colab_cli.common import state
from colab_cli.commands.session import resolve_runtime_options
from colab_cli.runtime import ColabRuntime
from colab_cli.state import SessionState
from colab_alerts import permitted,check
from colab_jobs import JOBS,atomic_json

def restore(session):
    jobs={"mind_r7_e5_eval":"e5_evaluation_queue","mind_r7_e5_fit_resume":"e5_refit_queue"}
    if session not in jobs:raise ValueError(session)
    permitted();d=JOBS/session
    if not (d/"machine.json").is_file() or (d/"released.json").is_file() or state.store.get(session):return
    data=[json.loads(v) for v in (Path.home()/".config/colab-cli/history"/(session+".jsonl")).read_text().splitlines()]
    original=next(v for v in reversed(data) if v["event_type"]=="session_created")
    endpoint=original["endpoint"];expected=json.loads((d/"machine.json").read_text())["uuid"]
    a=next((x for x in state.client.list_assignments() if x.endpoint==endpoint),None)
    if a is None:
        notebook=json.loads((d/"allocation.json").read_text())["notebook"]
        res=state.client._get_assignment(uuid.UUID(notebook),*resolve_runtime_options("A100",None,high_mem=original["machine_shape"]=="HIGH_RAM"))
        if isinstance(res,Assignment) and res.endpoint==endpoint:a=res
    if a is None:
        atomic_json(d/"mapping_lookup.json",dict(utc=time.time(),endpoint=endpoint,found=False));return
    info=a.runtime_proxy_info
    s=SessionState(name=session,endpoint=endpoint,url=info.url,token=info.token,token_expires_at=info.expires_at(),
                   variant=original["variant"],accelerator=original["accelerator"],machine_shape=original["machine_shape"])
    runtime=ColabRuntime(s.url,s.token,on_kernel_started=lambda kid:setattr(s,"kernel_id",kid),on_session_started=lambda sid:setattr(s,"session_id",sid))
    queue=jobs[session]
    query=f"""import json,subprocess
from pathlib import Path
r=Path('/content/mind/run');name={queue!r};pid=int((r/(name+'.pid')).read_text())
print('@@OWN '+json.dumps(dict(uuid=subprocess.check_output(['nvidia-smi','--query-gpu=uuid','--format=csv,noheader'],text=True).strip(),queue=json.loads((r/(name+'.status.json')).read_text())['job'],pid=pid,alive=Path('/proc',str(pid)).exists())))
"""
    output=runtime.execute_code(query,timeout=18)
    lines=[line for chunk in output for line in str(chunk.get("text","")).splitlines() if line.startswith("@@OWN ")]
    if not lines:raise RuntimeError("Ownership proof unavailable")
    proof=json.loads(lines[-1][len("@@OWN "):])
    if proof["uuid"]!=expected or proof["queue"]!=queue:raise ValueError("Remote identity did not match this project")
    if state.store.get(session) is None:state.store.add(s)
    atomic_json(d/"mapping_lookup.json",dict(utc=time.time(),endpoint=endpoint,found=True,rebound=True,proof=proof))

if __name__=="__main__":
    try:restore(sys.argv[1])
    except Exception as exc:
        check(str(exc),sys.argv[1],"restore project alias")
        print("Mapping check:",type(exc).__name__,file=sys.stderr);raise SystemExit(1)
