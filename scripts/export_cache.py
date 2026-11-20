import sys, time
from jobio import CACHE,RUN,EXPORT,wait_for,status,export_file,run_job

def fetch(kind):
    p=CACHE/(kind+".npy");wait_for(p)
    status("export_cache_"+kind,"exporting",size=p.stat().st_size)
    export_file(p)
    status("export_cache_"+kind,"DONE",size=p.stat().st_size)
if __name__=="__main__":
    kind=sys.argv[1]
    if kind not in ("qwen","bge"):raise ValueError(kind)
    run_job("export_cache_"+kind,lambda:fetch(kind))

