"""Local persistent alarms. A confirmed auth error halts all further project CLI calls."""
import fcntl
import json
from pathlib import Path
import re
import subprocess
import time

LOCAL=Path(__file__).resolve().parents[1]/"artifacts/local"
LATCH=LOCAL/"colab_auth_block.json"
AUTH=re.compile(r"authorization_response|invalid[_ ](?:grant|client|credentials)|expired_token|reauthentication|refresherror|unauthenticated|unauthorized|(?:http|status(?:_code)?)[^\n]{0,15}\b401\b|\b401\s+client error|access token (?:has )?expired|failed to refresh[^\n]*token|oauth[^\n]{0,75}(?:expired|fail|invalid|error)|authentication (?:required|failed)|login required|enter the authorization code|please visit this url to authorize|could not automatically determine credentials",re.I)
class AuthenticationBlocked(Exception):
    pass
def emit(kind,session,detail):
    LOCAL.mkdir(parents=True,exist_ok=True)
    with (LOCAL/"alert.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        obj={"type":kind,"session":session,"utc":time.time(),"message":detail}
        notify=True
        if kind=="authentication":
            notify=not LATCH.exists()
            if notify:
                tmp=LATCH.with_suffix(".new");tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2));tmp.replace(LATCH)
        if kind=="heartbeat":
            mark=LOCAL/(session+".heartbeat_alarm")
            notify=not mark.exists()
            if notify:mark.write_text(str(obj["utc"]))
        if kind not in ("authentication","heartbeat"):
            mark=LOCAL/(session+"."+kind+"_alarm")
            notify=not mark.exists() or time.time()-mark.stat().st_mtime>900
            if notify:mark.write_text(str(obj["utc"]))
        if not notify:return
        with (LOCAL/"ALERT").open("a") as out:out.write(json.dumps(obj,ensure_ascii=False)+"\n")
        script='on run argv\n display notification (item 1 of argv) with title "MIND Colab 告警" subtitle (item 2 of argv) sound name "Basso"\nend run'
        try:rc=subprocess.run(["/usr/bin/osascript","-e",script,detail,session],timeout=6,capture_output=True).returncode
        except (OSError,subprocess.TimeoutExpired):rc=-1
        with (LOCAL/"alert_notifications.jsonl").open("a") as out:
            out.write(json.dumps(dict(type=kind,session=session,utc=obj["utc"],osascript_exit_code=rc))+"\n")
def permitted():
    if LATCH.exists():raise AuthenticationBlocked("Colab authentication alarm is latched. See artifacts/local/ALERT.")
def check(output,session,operation):
    if AUTH.search(output):
        emit("authentication",session,f"Colab {operation} 认证失败，项目已停止继续发送 CLI 请求，请重新登录。")
        raise AuthenticationBlocked("Colab authentication failed; local alert and notification issued.")
