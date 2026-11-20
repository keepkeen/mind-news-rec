"""Invoke the installed Colab CLI with an idempotent notebook ID for this assignment only."""
import sys
import types
import uuid
from colab_cli.commands import session as session_cmd
from colab_cli.cli import main
allowed={"mind_fullfit_1119","mind_qfit_1119","mind_seed_eval_1119","mind_seed_fit_1119","mind_e5_eval_1119","mind_e5_fit_1119","mind_r7_e5_eval","mind_r7_e5_fit","mind_r7_bge","mind_r7_e5_fit_resume"}
name,notebook,gpu,*shape=sys.argv[1:]
if name not in allowed or gpu not in ("A100","H100"):raise ValueError(name)
identifier=uuid.UUID(notebook)
if name.startswith("mind_r7_"):
    import time
    from colab_cli.common import state
    from colab_cli.client import Assignment
    from colab_alerts import permitted,check as check_auth
    from colab_jobs import JOBS,atomic_json
    permitted()
    started=time.time()
    try:
        result=state.client._get_assignment(identifier,*session_cmd.resolve_runtime_options(gpu,None,high_mem=bool(shape)))
        if not isinstance(result,Assignment):
            atomic_json(JOBS/name/"billing_presence.json",dict(notebook=notebook,absent_at=started))
    except Exception as exc:check_auth(str(exc),name,"new")
# The stock 'new' command chooses a fresh notebook UUID. Reuse ours if a POST reply is lost.
session_cmd.uuid=types.SimpleNamespace(uuid4=lambda:identifier)
if shape not in ([],["high"]):raise ValueError(shape)
sys.argv=["colab","new","-s",name,"--gpu",gpu]+(["--high-mem"] if shape else [])
main()

