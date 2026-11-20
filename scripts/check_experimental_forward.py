"""Check future-window semantics with exact endpoints and direct scans of individual exposures."""
import sys
from types import SimpleNamespace
import numpy as np
from experimental_exposure import DAY, FORWARD, HOUR, WEEK, Timeline, values
from experimental_forward import WORK
from exposures import BASE, ITEM_BITS, ORIGIN, TIME_BITS, Reader
from ranking import arr
from trees import record

def reference(events, all_times, all_sizes, news, user, now, available):
    future = [(t,u) for t,u,n in events if n == news and now < t <= now + available]
    counts, rates, fractions = [], [], []
    for width in (HOUR,6*HOUR,DAY):
        observed = min(available,width)
        count = sum(t <= now+observed for t,_ in future)
        volume = sum(int(s[(t>now)&(t<=now+observed)].sum()) for t,s in zip(all_times,all_sizes))
        counts.append(np.log1p(count))
        rates.append(np.log1p(1e6*count/volume) if volume else 0.)
        fractions.append(observed/width)
    repeats = [t-now for t,u in future if u == user]
    elapsed = [t-now for t,_ in future]
    result = [*counts,*rates,bool(repeats),np.log1p(len(repeats)),np.log1p(min(repeats,default=0)),
              np.log1p(max(elapsed,default=0)),*fractions,available/WEEK]
    return np.asarray(result,np.float32)

def fixture():
    t = ORIGIN+20
    offsets = [-1,0,0,1,HOUR-1,HOUR,HOUR+1,6*HOUR,6*HOUR+1,DAY,DAY+1,WEEK,WEEK+1]
    events = [(t+d,3 if j%2==0 else 4,1) for j,d in enumerate(offsets)]
    events += [(t,3,2),(t+HOUR,4,2),(t+WEEK+1,3,2)]
    et = np.array([v[0] for v in events])
    users = np.array([v[1] for v in events])
    news = np.array([v[2] for v in events])
    ak = np.sort((news<<TIME_BITS) + et-ORIGIN)
    pk = np.sort((((users<<ITEM_BITS)+news)<<TIME_BITS) + et-ORIGIN)
    r = SimpleNamespace(off=np.array([0,2]),cand=np.array([1,2]),t=np.array([t]),
                        u=np.array([0]),uid=np.array([3]),article=ak,ua=pk,
                        ap=np.searchsorted(ak,np.arange(4)<<TIME_BITS),
                        up=np.searchsorted(pk,np.arange(6)<<(TIME_BITS+ITEM_BITS)))
    timeline = SimpleNamespace(traffic=np.r_[0,np.bincount(et-ORIGIN).cumsum()])
    cases = 0
    for horizon in (0,1,HOUR-1,HOUR,6*HOUR,DAY-1,DAY,WEEK):
        actual = values(r,np.array([0]),np.array([horizon],np.int32),timeline)
        for j,n in enumerate((1,2)):
            expected = reference(events,[et],[np.ones(len(et),int)],n,3,t,horizon)
            np.testing.assert_allclose(actual[j],expected,rtol=1e-7,atol=1e-7)
            cases += 1
    return cases

def real_examples():
    tl=Timeline()
    names=("train","dev","test")
    readers={s:Reader(s) for s in names}
    sizes=[np.diff(readers[s].off) for s in names]
    ts=[readers[s].t for s in names]
    details=[]
    for split,selector in (("train","cross_split"),("dev","cross_split"),("test","last_hour"),("test","last_second")):
        r=readers[split]
        horizon=tl.horizons(split)
        if selector=="cross_split":
            following=readers[names[names.index(split)+1]]
            lower,upper=int(following.t.min()),int(following.t.max())
            group=np.flatnonzero((horizon>DAY)&(r.t>=r.t.max()-HOUR))
            for row in group[np.argsort(-r.t[group],kind="stable")[:120]]:
                candidates=r.cand[r.off[row]:r.off[row+1]]
                def cross_count(news):
                    key=(int(news)<<TIME_BITS)-ORIGIN
                    times=r.article[r.ap[news]:r.ap[news+1]]
                    right=min(upper,int(r.t[row])+int(horizon[row]))
                    return int(np.searchsorted(times,key+right,side="right")-np.searchsorted(times,key+lower))
                cross=[cross_count(n) for n in candidates]
                j=int(np.argmax(cross))
                if cross[j]>0:break
            else:raise ValueError("No following-split candidate in the inspected slates")
        elif selector=="last_hour":
            row=np.argmin(np.abs(r.t.astype(np.int64)-(tl.end-1800)))
        else:
            row=np.argmax(r.t)
        candidates=r.cand[r.off[row]:r.off[row+1]]
        if selector!="cross_split":j=int(np.argmax(r.ap[candidates+1]-r.ap[candidates]))
        n=int(candidates[j]); u=int(r.uid[r.u[row]]); t=int(r.t[row]); available=int(horizon[row])
        observed_by_split={};events=[]
        for s in names:
            source=readers[s]
            positions=np.flatnonzero(source.cand==n)
            rows=np.searchsorted(source.off[1:],positions,side="right")
            tt=source.t[rows]; uu=source.uid[source.u[rows]]
            valid=(tt>t)&(tt<=t+available)
            observed_by_split[s]=int(valid.sum())
            events.extend((int(a),int(b),n) for a,b in zip(tt[valid],uu[valid]))
        expected=reference(events,ts,sizes,n,u,t,available)
        actual=values(r,np.array([row]),horizon,tl)[j]
        np.testing.assert_allclose(actual,expected,rtol=1e-7,atol=1e-7)
        if selector=="cross_split":
            next_split=names[names.index(split)+1]
            if observed_by_split[next_split]==0:raise ValueError(f"Missing cross-boundary audit example for {split}")
        details.append(dict(split=split,case=selector,impression_id=int(arr(BASE,split,"imp_id")[row]),news=n,
            user=u,t=t,observed_seconds=available,strictly_future_by_split=observed_by_split,
            max_abs_error=float(np.max(np.abs(actual-expected))),
            features={c:float(v) for c,v in zip(FORWARD,actual)}))
    return details

if __name__=="__main__":
    count=fixture()
    if "--fixture-only" in sys.argv:
        print("Passed endpoint fixtures",count)
    else:
        cases=real_examples()
        record(WORK/"feature_audit.json",dict(synthetic_candidate_cases=count,raw_cross_checks=cases))
        print("Passed",count,"synthetic candidate cases and",len(cases),"real slate candidates.")
