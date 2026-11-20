"""Stop only this checkout's detached experiment services. Never contact a remote runtime."""
import argparse
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import time
from colab_jobs import ART,NAMES,ROOT,atomic_json

SERVICES={"colab_jobs.py","release_gpu.py","project_budget.py","e5_admission.py","asym_admission.py",
          "fleet_boot.py","e5_launch.py","coordinate.py","incremental.py"}

def cwd(pid):
    p=subprocess.run([shutil.which("lsof") or "/usr/sbin/lsof","-a","-p",str(pid),"-d","cwd","-Fn"],
                     capture_output=True,text=True,timeout=5)
    return next((Path(v[1:]).resolve() for v in p.stdout.splitlines() if v.startswith("n")),None)

def inventory():
    pidfiles={}
    for p in (ART/"local").glob("*.pid"):
        try:pidfiles.setdefault(int(p.read_text().strip()),[]).append(p.name)
        except ValueError:continue
    lines=subprocess.check_output(["ps","-axo","pid=,ppid=,pgid=,command="],text=True).splitlines()
    rows={}
    for line in lines:
        fields=line.strip().split(None,3)
        if len(fields)!=4:continue
        pid,parent,group=map(int,fields[:3]);rows[pid]=(parent,group,fields[3])
    own=[]
    for pid,(parent,group,cmd) in rows.items():
        if pid!=group or pid not in pidfiles:continue
        if any("r7_" in name for name in pidfiles[pid]):continue
        try:argv=shlex.split(cmd)
        except ValueError:continue
        if len(argv)<2 or not Path(argv[0]).name.startswith("python"):continue
        sp=Path(argv[1])
        if sp.name not in SERVICES or cwd(pid)!=ROOT or (ROOT/sp).resolve()!=ROOT/"scripts"/sp.name:continue
        args=argv[2:]
        if any(v.startswith("mind_r7_") for v in args):continue
        if sp.name=="colab_jobs.py" and (not args or args[0]!="monitor" or not set(args).intersection(NAMES)):continue
        if sp.name=="release_gpu.py" and (not args or args[0] not in NAMES):continue
        if sp.name in ("project_budget.py","incremental.py") and args!=["watch"]:continue
        if sp.name=="coordinate.py" and args:continue
        members=[i for i,(_,g,_) in rows.items() if g==group]
        # A detached service owns its private process group, including its caffeinate helper and CLI children.
        own.append(dict(pid=pid,group=group,service=sp.name,args=args,registered=pidfiles[pid],members=members))
    return sorted(own,key=lambda v:v["pid"])

def main(apply):
    found=inventory()
    if not apply:
        print("Active project services",[(r["pid"],r["service"]) for r in found]);return
    d=ART/"round7";d.mkdir(parents=True,exist_ok=True)
    if not found:
        print("Active project services: 0");return
    # Snapshot the uncorrected ledger before stopping the poller that maintained it.
    original=d/"budget_before_cleanup.json"
    if not original.exists():shutil.copy2(ART/"fleet/budget.json",original)
    for r in found:
        try:
            if os.getpgid(r["pid"])==r["group"] and cwd(r["pid"])==ROOT:os.killpg(r["group"],signal.SIGTERM)
        except ProcessLookupError:pass
    for _ in range(15):
        survivors=inventory()
        if not survivors:break
        time.sleep(.2)
    if survivors:raise RuntimeError(f"Unstopped project services {survivors}")
    receipt=dict(stopped_utc=time.time(),groups=found,active_after=survivors,remote_calls=0)
    atomic_json(d/"local_cleanup.json",receipt)
    print("Stopped",len(found),"project process groups; active after cleanup",len(survivors))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--apply",action="store_true")
    main(p.parse_args().apply)
