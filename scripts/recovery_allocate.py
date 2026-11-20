"""Retry only temporary capacity failures on one idempotent recovery notebook. Auth errors exit immediately."""
import argparse
import fcntl
import os
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from colab_alerts import permitted
from colab_jobs import ART,JOBS,atomic_json
from fleet_boot import boot
from recovery_watchdog import SESSIONS

def main(s):
    d=JOBS/s;d.mkdir(parents=True,exist_ok=True)
    cut=datetime(2026,11,19,23,58,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp() if "e5" in s else datetime(2026,11,20,2,45,tzinfo=ZoneInfo("Asia/Shanghai")).timestamp()
    with (d/"retry.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        n=0
        while not (d/"machine.json").exists() and time.time()<cut and not (ART/"round7/no_new_assignments").exists():
            if s.endswith("_bge") and (ART/"round7/stop_bge").exists():return
            permitted()
            high=(n%2==0) if "e5" in s else (n%2==1)
            atomic_json(d/"retry_state.json",dict(attempt=n+1,shape="high" if high else "standard",utc=time.time(),cutoff=cut))
            try:boot(s,"A100",1,"high" if high else None)
            except BlockingIOError:time.sleep(5);continue
            if (d/"machine.json").exists():return
            n+=1;time.sleep(min(90,20*n))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("session",choices=SESSIONS);main(p.parse_args().session)
