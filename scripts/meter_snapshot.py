"""Capture a read-only account rate and the hardware inventory needed to identify the project's marginal rate."""
import hashlib
import json
import sys
import time
from colab_cli.common import state
from colab_jobs import NAMES,ART,atomic_json

def snapshot():
    names={s.endpoint:s.name for s in state.store.list().values()}
    def inventory():
        own={};other=[];unknown=0
        for a in state.client.list_assignments():
            shape=(a.accelerator.value,a.machine_shape.name)
            name=names.get(a.endpoint)
            if name in NAMES:own[name]=shape
            else:other.append((a.endpoint,*shape));unknown+=name is None
        digest=hashlib.sha256(json.dumps(sorted(other)).encode()).hexdigest()
        return own,digest,len(other),unknown
    a=inventory();rate=state.client.get_consumption_user_info();b=inventory()
    if a!=b or rate.assignments_count!=len(a[0])+a[2]:raise RuntimeError("Assignment set changed during meter snapshot")
    return dict(utc=time.time(),own=a[0],other_fingerprint=a[1],other_count=a[2],untracked=a[3],
                active=rate.assignments_count,account_cu_per_hour=rate.consumption_rate_hourly)
if __name__=="__main__":
    name=sys.argv[1]
    if name not in ("before","after"):raise ValueError(name)
    out=snapshot();atomic_json(ART/"round6"/("meter_"+name+".json"),out)
    print("tracked project devices",len(out["own"]),"account count",out["active"],"CU/h",out["account_cu_per_hour"])
