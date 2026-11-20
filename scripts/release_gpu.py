"""Stop a single named project A100 once its queued work and local artifact synchronization finish."""
import argparse
import json
from pathlib import Path
import time
from colab_alerts import LATCH
from colab_jobs import NAMES,JOBS,ART,atomic_json,call,rex,response,completed_eval_epochs,omitted_test_cache

ROLES={"mind_h100":"fullbge","mind_plm_1119":"qlora"}
TERMINAL=("DONE","FAILED","SKIPPED")
def released(session):
    return JOBS/session/"released.json"

def main(session,terminal_after=0.,queues=()):
    d=JOBS/session
    queues=tuple(queues) or ("queue_"+ROLES[session],)
    if any(not q.replace("_","").isalnum() for q in queues):raise ValueError(queues)
    if released(session).exists():return
    prog=d/"release_progress.json"
    while True:
        if LATCH.is_file() or released(session).exists():return
        hb=d/"heartbeat.json"
        if hb.exists():
            states=json.loads(hb.read_text())["status"]
            if all(states.get(q+".status.json",{}).get("phase") in TERMINAL and
                   states[q+".status.json"].get("utc_epoch",0)>terminal_after for q in queues):break
        time.sleep(45)
    # Confirm ownership on the live VM as well as the local session-name allowlist.
    code=f"""
from pathlib import Path
import json,os
r=Path('/content/mind');phases={{}}
for name in {queues!r}:
    q=r/'run'/(name+'.status.json')
    assert (r/'run'/(name+'.pid')).exists() and q.exists()
    phases[name]=json.loads(q.read_text())['phase']
    assert phases[name] in {TERMINAL!r}
for f in (r/'run').glob('*.log'):
    dst=r/'export'/(f.name+'.txt')
    if not dst.exists() or dst.stat().st_mtime < f.stat().st_mtime:
        with f.open('rb') as lf:
            lf.seek(max(0,f.stat().st_size-300000));tmp=dst.with_suffix('.tmp');tmp.write_bytes(lf.read());tmp.replace(dst)
exports=r/'export';parts=[];small=[]
for f in exports.iterdir():
    if f.name.endswith('.parts.json'):
        m=json.loads(f.read_text());parts.append([m['name'],m['sha256']])
    elif '.part' not in f.name and not f.name.endswith('.tmp'):
        small.append([f.name,f.stat().st_size,f.stat().st_mtime_ns])
print('@@MIND '+json.dumps({{'phase':phases,'parts':parts,'small':small}}))
"""
    manifest=response(rex(session,code,70,3))
    while True:
        if LATCH.is_file() or released(session).exists():return
        selected=completed_eval_epochs()
        retired=json.loads((d/"rolling_superseded.json").read_text()) if (d/"rolling_superseded.json").exists() else {}
        terminal_optional=[]
        if session.startswith("mind_r7_") and all(v=="DONE" for v in manifest["phase"].values()):
            from recovery_sync import ROLL
            # A completed queue can release as soon as its permanent weights and scores are verified.
            terminal_optional=[n for n,_ in manifest["parts"] if ROLL.fullmatch(n) and not (d/(n+".verified")).exists()]
        ledger=json.loads((d/"sync.json").read_text()) if (d/"sync.json").exists() else {}
        missing=[n for n,sha in manifest["parts"] if n not in retired and n not in terminal_optional and not omitted_test_cache(n,selected) and not ((d/n).is_file() and (d/(n+".verified")).is_file()
                                                        and (d/(n+".verified")).read_text().strip()==sha)]
        for n,size,stamp in manifest["small"]:
            dest=d/n
            # Logs are mutable snapshots; once the queue is terminal the final byte lengths stabilize.
            if not dest.is_file() or dest.stat().st_size!=size or (not n.endswith(".log.txt") and ledger.get(n)!=[size,stamp]):
                missing.append(n)
        atomic_json(prog,dict(stage="syncing_before_stop",pending=len(missing),sample=missing[:4],phase=manifest["phase"]))
        if not missing:break
        time.sleep(18)
    optional=[n for n,_ in manifest["parts"] if omitted_test_cache(n,selected)]
    info=dict(session=session,queues=queues,queue_phase=manifest["phase"],
              verified_large=sum(n not in retired and n not in optional and n not in terminal_optional for n,_ in manifest["parts"]),
              superseded_rolling=list(retired),terminal_optional_rolling=terminal_optional,
              optional_reproducible_test_caches=optional,synced_small=len(manifest["small"]))
    # Do not let an uncertain HTTP reply trigger an operation on any other session.
    try:out=call(["stop","-s",session],95,3)
    except RuntimeError as e:
        if "not found" not in str(e).lower():raise
        out="Named session already absent"
    (d/"stop_monitor").touch()
    info.update(stopped_utc=time.time(),response=out[-250:]);atomic_json(released(session),info)
    atomic_json(prog,dict(stage="STOPPED",**info))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("session",choices=NAMES)
    p.add_argument("--terminal-after",type=float,default=0.)
    p.add_argument("--queues",nargs="+")
    args=p.parse_args()
    while True:
        try:
            main(args.session,args.terminal_after,args.queues or ())
            break
        except (RuntimeError,OSError,ValueError) as exc:
            atomic_json(JOBS/args.session/"release_progress.json",
                        dict(stage="retrying_sync_or_stop",error=str(exc)[-430:],time=time.time()))
            time.sleep(40)

