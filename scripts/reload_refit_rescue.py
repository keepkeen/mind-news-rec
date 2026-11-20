"""Reload the MIND local services for the replacement fit without interrupting tree workers."""
import concurrent.futures
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from colab_jobs import ART,ROOT,atomic_json

def ps(pid,field):
    p=subprocess.run(["ps","-p",str(pid),"-o",field+"="],capture_output=True,text=True)
    return p.stdout.strip() if p.returncode==0 else ""

def reload_one(name,old,new,wait_children=False):
    target=ART/"local"/(name+".pid")
    pid=int(target.read_text()) if target.is_file() else 0
    command=ps(pid,"command") if pid else ""
    if command:
        argv=shlex.split(command)
        cwd=subprocess.check_output(["/usr/sbin/lsof","-a","-p",str(pid),"-d","cwd","-Fn"],text=True).splitlines()
        if (ROOT/argv[0]).resolve()!=Path(sys.executable).resolve() or os.getpgid(pid)!=pid or "n"+str(ROOT) not in cwd or argv[1:] not in (old,new):
            raise RuntimeError(f"Unexpected owner/command: {name} {pid}")
        if wait_children:
            while True:
                os.kill(pid,signal.SIGSTOP)
                children=subprocess.check_output(["ps","-axo","ppid=,stat=,command="],text=True).splitlines()
                work=[x for l in children if (x:=l.strip().split(maxsplit=2))[0]==str(pid) and not x[1].startswith("Z") and "/usr/bin/caffeinate " not in x[2]]
                if not work:break
                os.kill(pid,signal.SIGCONT);time.sleep(.5)
        os.killpg(pid,signal.SIGTERM)
        if wait_children:
            try:os.kill(pid,signal.SIGCONT)
            except ProcessLookupError:pass
        limit=time.monotonic()+5
        while ps(pid,"command") and time.monotonic()<limit:time.sleep(.05)
        if ps(pid,"command"):raise RuntimeError("Local process did not exit: "+name)
    result=dict(old_pid=pid,utc=time.time(),old=old,new=new)
    if new:
        subprocess.run([sys.executable,"scripts/detach.py",name,"/usr/bin/caffeinate","-i",sys.executable,*new],cwd=ROOT,check=True)
        result["new_pid"]=int(target.read_text())
    return name,result

if __name__=="__main__":
    before=["scripts/release_gpu.py","mind_r7_e5_eval","--queues","e5_evaluation_queue"]
    base=[("r7_alerts_live",["scripts/recovery_watchdog.py"]),("r7_guard_live",["scripts/recovery_budget.py","--guard"]),
          ("r7_priority_sync_mind_r7_e5_eval",["scripts/colab_jobs.py","monitor","-s","mind_r7_e5_eval"]),
          ("r7_rescue_release",["scripts/release_gpu.py","mind_r7_e5_fit_resume","--queues","e5_refit_queue"])]
    specs=[(n,k,k,False) for n,k in base]
    specs += [("r7_verified_release_mind_r7_e5_eval",before,before+["e5_trainonly_replay"],False),
              ("r7_verified_release_mind_r7_e5_fit",["scripts/release_gpu.py","mind_r7_e5_fit","--queues","e5_refit_queue"],None,False),
              ("r7_lock_coordinator",["scripts/coordinate.py"],["scripts/coordinate.py"],True),
              ("r7_incremental",["scripts/incremental.py","watch"],["scripts/incremental.py","watch"],True)]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result=dict(pool.map(lambda args:reload_one(*args),specs))
    atomic_json(ART/"round7/rescue_local_services.json",result)
