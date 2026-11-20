"""Show explicitly how train and dev exposures enter a real test candidate's strict time prefix."""
import numpy as np
from numba import njit
from colab_jobs import ART,atomic_json
from exposures import Reader,BASE,PREFIX
from ranking import arr

@njit
def at_time(off,ids,times,news,cut):
    before=equal=after=0
    for i in range(len(times)):
        for j in range(off[i],off[i+1]):
            if ids[j]!=news:continue
            if times[i]<cut:before+=1
            elif times[i]==cut:equal+=1
            else:after+=1
    return before,equal,after
def main():
    r=Reader("test");scans={}
    for split in ("train","dev"):
        count=np.zeros(len(np.load(BASE/"cat.npy")),np.int64);cand=arr(BASE,split,"candidate")
        for j in range(0,len(cand),1000000):count+=np.bincount(cand[j:j+1000000],minlength=len(count))
        scans[split]=count
    first=np.argpartition(r.t,10000)[:10000]
    for i in first[np.argsort(r.t[first],kind="stable")]:
        candidates=r.cand[r.off[i]:r.off[i+1]]
        valid=np.flatnonzero((scans["train"][candidates]>0)&(scans["dev"][candidates]>0))
        if len(valid):break
    else:raise ValueError("No mixed-period candidate")
    j=int(valid[0]);news=int(candidates[j]);cut=int(r.t[i]);events={};bounds={}
    for split in ("train","dev","test"):
        ts=arr(BASE,split,"time");bounds[split]=[int(ts.min()),int(ts.max())]
        v=at_time(arr(BASE,split,"offset"),arr(BASE,split,"candidate"),ts,news,cut)
        events[split]=dict(zip(("strictly_before","same_second","later"),map(int,v)))
    total=sum(v["strictly_before"] for v in events.values());observed=float(r.prefix(np.array([i]))[j,0])
    expected=float(np.float32(np.log1p(total)))
    if observed!=expected or not events["test"]["same_second"] or not bounds["train"][1]<bounds["dev"][0] or not bounds["dev"][1]<bounds["test"][0]:raise ValueError((events,observed,expected))
    out=dict(test_row=int(i),impression_id=int(arr(BASE,"test","imp_id")[i]),dense_news_id=news,epoch=cut,
        counts=events,strict_prior_count=total,feature=PREFIX[0],actual=observed,expected=expected,time_ranges=bounds)
    atomic_json(ART/"round7/prefix_boundary.json",out);print(out)
if __name__=="__main__":main()
