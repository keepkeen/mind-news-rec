"""Watch fresh MIND heartbeats even if the Colab clients, kernels or authentication stop responding."""
import json
import time
from colab_jobs import ART,JOBS,atomic_json
from colab_alerts import LATCH,LOCAL,AUTH,AuthenticationBlocked,check as check_auth,emit

SESSIONS=("mind_r7_e5_eval","mind_r7_e5_fit","mind_r7_bge")
def current_roles():
    return (SESSIONS[0],"mind_r7_e5_fit_resume",SESSIONS[2]) if (ART/"round7/e5_fit_incident.json").exists() else SESSIONS
def check(now=None):
    now=time.time() if now is None else now
    states={}
    for s in current_roles():
        d=JOBS/s;m=d/"machine.json"
        if (d/"released.json").exists():states[s]={"state":"released"};continue
        if not m.exists():states[s]={"state":"unallocated"};continue
        start=json.loads(m.read_text())["allocated_utc"]
        hb=d/"heartbeat.json";h=json.loads(hb.read_text()).get("timestamp",0) if hb.exists() else 0
        age=now-max(start,h);states[s]={"state":"active","last_success":h,"age_seconds":age}
        if age>=900:emit("heartbeat",s,f"MIND 心跳已中断 {int(age/60)} 分钟，请检查认证与会话。")
        else:(LOCAL/(s+".heartbeat_alarm")).unlink(missing_ok=True)
    return states
def main():
    offsets_file=ART/"round7/alert_log_offsets.json"
    offsets=json.loads(offsets_file.read_text()) if offsets_file.is_file() else {}
    while True:
        if not LATCH.exists():
            for path in LOCAL.glob("*.log"):
                if "r7_" not in path.name:continue
                size=path.stat().st_size;start=offsets.get(path.name,0)
                if start==size:continue
                with path.open("rb") as f:
                    f.seek(max(0,size-131072,start if start<size else 0));data=f.read().decode(errors="replace")
                offsets[path.name]=size
                if AUTH.search(data):
                    try:check_auth(data,path.stem,"background CLI")
                    except AuthenticationBlocked:break
            atomic_json(offsets_file,offsets)
        states=check();atomic_json(ART/"round7/alerts_status.json",dict(utc=time.time(),auth_latched=LATCH.exists(),sessions=states))
        if all(x["state"]=="released" for x in states.values()):return
        time.sleep(10)
if __name__=="__main__":main()
