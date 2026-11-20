"""Run a repo-local command independently of the caller, with a pid and log."""
import os, subprocess, sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
name,*cmd=sys.argv[1:]
if not name.replace("_","").isalnum() or not cmd: raise ValueError("name command [args]")
log=root/"artifacts/local"/(name+".log")
pidpath=log.with_suffix(".pid")
if pidpath.exists():
    pid=int(pidpath.read_text())
    try: os.kill(pid,0)
    except ProcessLookupError: pass
    else: raise SystemExit("Already running: "+str(pid))
with log.open("ab",buffering=0) as f:
    proc=subprocess.Popen(cmd,cwd=root,env=dict(os.environ,PYTHONUNBUFFERED="1"),stdin=subprocess.DEVNULL,
                          stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
pidpath.write_text(str(proc.pid))
print(name,proc.pid,log)

