import concurrent.futures as cf
import json
import subprocess
import sys
import time
from pathlib import Path
from colab_jobs import ART,JOBS,ROOT,deploy,upload_group,rex,response

def cmd(*argv):
    subprocess.run([sys.executable,*argv],cwd=ROOT,check=True)

def main():
    sess="mind_plm_1119"
    start=time.time()
    while not (JOBS/sess/"machine.json").exists():
        if time.time()-start>690:
            # First GPU is usable and the two frozen branches have bounded runtimes.
            sess="mind_h100";break
        time.sleep(7)
    if sess=="mind_plm_1119":
        cmd("scripts/detach.py","monitor_plm","/usr/bin/caffeinate","-i",sys.executable,"scripts/colab_jobs.py","monitor","-s",sess)
        with cf.ThreadPoolExecutor(max_workers=2) as pool:
            dep=pool.submit(deploy,sess,["jobio.py","ranking.py","recommender.py","upper.py"])
            upl=pool.submit(lambda:[upload_group(sess,g) for g in ("shared","train","dev","test")])
            dep.result()
            cmd("scripts/colab_jobs.py","start","-s",sess,"upper","recommender.py","upper","2","192","0.0005","1122")
            upl.result()
    else:
        print("Single-GPU fallback after Qwen",flush=True)
        while True:
            hb=JOBS/sess/"heartbeat.json"
            if hb.exists():
                rows=json.loads(hb.read_text())["status"]
                if rows.get("followup.status.json",{}).get("phase") in ("DONE","FAILED"):break
            time.sleep(45)
        cmd("scripts/colab_jobs.py","start","-s",sess,"upper","recommender.py","upper","2","192","0.0005","1122")
    (ART/"local/plm_session.txt").write_text(sess)
if __name__=="__main__":main()

