"""Independently rebuild selected features from earlier raw-order compact slates without exposure indices."""
import numpy as np
from numba import njit
from exposures import BASE,Reader,PREFIX
from ranking import arr
from trees import record

@njit(cache=True)
def count_events(off,cand,times,users,uids,selected,requested,cutoffs,owners,counts,first,last_personal):
    maximum=np.max(cutoffs)
    for i in range(len(times)):
        now=times[i]
        if now>=maximum:continue
        for j in range(off[i],off[i+1]):
            news=cand[j]
            if not selected[news]:continue
            for q in range(len(cutoffs)):
                if requested[q]!=news or now>=cutoffs[q]:continue
                dt=cutoffs[q]-now
                counts[q,0]+=1;counts[q,1]+=dt<=3600;counts[q,2]+=dt<=21600;counts[q,3]+=dt<=86400
                if now<first[q]:first[q]=now
                if owners[q]==uids[users[i]]:
                    counts[q,4]+=1
                    if now>last_personal[q]:last_personal[q]=now

def main():
    expected=[];queries=[];description=[]
    for split in ("train","dev","test"):
        r=Reader(split)
        for row in sorted(set([int(np.argmin(r.t)),0,17,len(r.u)//2,len(r.u)-1])):
            x=r.prefix(np.array([row]))
            # Include previously exposed and repeated candidates where present.
            for col in sorted(set((0,int(x[:,5].argmax()),int(x[:,0].argmax())))):
                expected.append(x[col,:len(PREFIX)])
                queries.append([r.cand[r.off[row]+col],r.t[row],r.uid[r.u[row]]])
                description.append([split,row,col])
    news,t,owner=np.asarray(queries,dtype=np.int64).T
    selected=np.zeros(len(np.load(BASE/"cat.npy")),bool);selected[news]=True
    counts=np.zeros((len(t),5),np.int64);first=np.full(len(t),2**40,dtype=np.int64);last=np.zeros(len(t),np.int64)
    for split in ("train","dev","test"):
        args=[arr(BASE,split,k) for k in ("offset","candidate","time","user","user_id")]
        count_events(*args,selected,news,t,owner,counts,first,last)
    n,p1,p6,p24,usern=counts.T
    brute=np.column_stack((np.log1p(n),np.log1p(p1),np.log1p(p6),np.log1p(p24),
        np.log1p(np.where(n>0,t-first,0)),np.log1p(usern),np.log1p(np.where(usern>0,t-last,0)),
        usern>0,np.log((p1+2)/((p6-p1)/5.+2)))).astype(np.float32)
    err=np.abs(brute-np.asarray(expected))
    np.testing.assert_allclose(brute,expected,rtol=5e-7,atol=5e-7)
    result=dict(queries=len(t),matched_features=len(PREFIX),max_abs_error=float(err.max()),cases=description,
                already_seen_personally=int((usern>0).sum()),strict_cutoff="event timestamp < scoring timestamp")
    record("artifacts/fusion/real_prefix_audit.json",result);print({k:v for k,v in result.items() if k!="cases"})
if __name__=="__main__":main()

