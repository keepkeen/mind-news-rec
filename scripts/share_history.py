"""Copy the verified frozen 8B history cache to the two optional runtimes, in bounded chunks."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import os
from pathlib import Path
import tempfile
import sys
from colab_jobs import ART,JOBS,CHUNK,REMOTE,atomic_json,call,rex,response

EXPECTED="f7588663b9d9f4fd44cc740739b48fd43bc320988e9e1b505ac7f4a5fd1739e2"
def copy(session):
    if session not in ("mind_seed_eval_1119","mind_seed_fit_1119"):raise ValueError(session)
    source=JOBS/"mind_h100/qwen.npy"
    if source.with_name(source.name+".verified").read_text().strip()!=EXPECTED:raise ValueError("Unverified history cache")
    ck=JOBS/session/"history_copy.json"
    size=source.stat().st_size
    status=response(rex(session,"""from pathlib import Path
import json
r=Path('/content/mind')
print('@@MIND '+json.dumps({'done':(r/'cache/qwen.sha256').read_text().strip() if (r/'cache/qwen.sha256').exists() else '', 'parts':{p.name:p.stat().st_size for p in (r/'incoming').glob('history.[0-9][0-9][0-9][0-9]')}}))""",55))
    if status["done"]==EXPECTED:return
    blocks=(size+CHUNK-1)//CHUNK
    atomic_json(ck,dict(stage="upload",blocks=blocks))
    with tempfile.TemporaryDirectory(prefix="history_upload_",dir=ART/"local") as tmp:
        def send(i):
            name=f"history.{i:04d}";length=min(CHUNK,size-i*CHUNK)
            if status["parts"].get(name)==length:return
            f=Path(tmp)/name
            with source.open("rb") as fp:fp.seek(i*CHUNK);f.write_bytes(fp.read(length))
            try:call(["upload","-s",session,str(f),REMOTE+"/incoming/"+name],115,4)
            finally:f.unlink(missing_ok=True)
        with ThreadPoolExecutor(max_workers=4) as pool:list(pool.map(send,range(blocks)))
    cmd=f"""import hashlib,pathlib,json
r=pathlib.Path('/content/mind');dest=r/'cache/qwen.npy';tmp=dest.with_suffix('.copying')
h=hashlib.sha256()
with tmp.open('wb') as f:
 for i in range({blocks}):
  b=(r/'incoming'/('history.%04d'%i)).read_bytes();h.update(b);f.write(b)
assert h.hexdigest()=={EXPECTED!r} and tmp.stat().st_size=={size}
tmp.replace(dest);(r/'cache/qwen.sha256').write_text(h.hexdigest())
for i in range({blocks}):(r/'incoming'/('history.%04d'%i)).unlink(missing_ok=True)
print('@@MIND '+json.dumps({{'stage':'READY','sha256':h.hexdigest(),'bytes':dest.stat().st_size}}))
"""
    atomic_json(ck,response(rex(session,cmd,90,3)))
if __name__=="__main__":copy(sys.argv[1])
