"""Diagnostic only: full-week exposure features are confined to the labeled Nov 9–14 window."""
import argparse
from pathlib import Path
import time
import numpy as np
from numba import njit
from exposures import Reader,ORIGIN,TIME_BITS,ITEM_BITS,search,PREFIX
from trees import fit_tree,record,row_metrics,summarized,zscore

HERE=Path("artifacts/measurement")
FUTURE=("log_week_exposure","log_remaining_exposure","log_to_last","log_week_user_exposure",
        "log_remaining_user_exposure","shown_later_to_user","log_to_next_user_exposure")
def ts(d):
    return int(np.datetime64("2019-11-"+str(d)+"T00:00:00").astype("int64"))

@njit(cache=True)
def lookahead(rows,off,cand,times,uids,users,article,ap,ua,up,stop):
    out=np.empty((np.sum(off[rows+1]-off[rows]),7),np.float32)
    at=0
    for i in rows:
        t=np.int64(times[i])-ORIGIN;u=np.int64(uids[users[i]])
        for j in range(off[i],off[i+1]):
            n=np.int64(cand[j]);start=n<<TIME_BITS
            end=search(article,ap[n],ap[n+1],start+stop)
            future=search(article,ap[n],end,start+t+1)
            pair=((u<<ITEM_BITS)+n)<<TIME_BITS
            left=search(ua,up[u],up[u+1],pair)
            right=search(ua,left,up[u+1],pair+stop)
            nxt=search(ua,left,right,pair+t+1)
            out[at,0]=np.log1p(end-ap[n]);out[at,1]=np.log1p(end-future)
            out[at,2]=np.log1p(max(0,article[end-1]-start-t))
            out[at,3]=np.log1p(right-left);out[at,4]=np.log1p(right-nxt)
            out[at,5]=right>nxt;out[at,6]=np.log1p(ua[nxt]-pair-t) if nxt<right else 0.
            at+=1
    return out

def data(r,rows):
    x=r.prefix(rows)[:,:len(PREFIX)]
    f=lookahead(rows,r.off,r.cand,r.t,r.uid,r.u,r.article,r.ap,r.ua,r.up,ts(15)-ORIGIN)
    return x,np.column_stack([x,f])

def main(args):
    HERE.mkdir(parents=True,exist_ok=True)
    prog=HERE/"progress.json"
    r=Reader("train");rng=np.random.default_rng(1119)
    pool=np.flatnonzero((r.t>=ts(12))&(r.t<ts(13)))
    chosen=np.sort(rng.choice(pool,min(args.fit_groups,len(pool)),replace=False))
    sizes=np.diff(r.off)[chosen];idx=r.positions(chosen);labels=r.labels[idx]
    record(prog,dict(stage="fit_exposure",groups=len(chosen),candidates=len(idx)))
    x,xf=data(r,chosen)
    base_names=("prefix","prefix_future")
    models={}
    for name,mat,names in zip(base_names,(x,xf),(PREFIX,(*PREFIX,*FUTURE))):
        models[name]=fit_tree(mat,labels,sizes,names,args.rounds)
        models[name].save_model(str(HERE/(name+".lgb")))
    shadow=Path("artifacts/jobs/mind_h100/bge_shadow_e2_train.npy")
    record(prog,dict(stage="waiting_for_train_only_content",path=str(shadow),groups=len(chosen)))
    stop=time.time()+5*3600
    while not shadow.with_name(shadow.name+".verified").exists():
        if time.time()>stop: raise TimeoutError(shadow)
        time.sleep(15)
    content=np.load(shadow,mmap_mode="r")
    if len(content)!=r.off[-1]: raise ValueError("Misaligned content logits")
    init=zscore(content[idx],sizes)
    for name,mat,names in zip(base_names,(x,xf),(PREFIX,(*PREFIX,*FUTURE))):
        key="content_"+name
        models[key]=fit_tree(mat,labels,sizes,names,args.rounds,init=init)
        models[key].save_model(str(HERE/(key+".lgb")))
    del x,xf,init
    totals={};results={};names=(*models.keys(),"content")
    for day in (13,14):
        rows=np.flatnonzero((r.t>=ts(day))&(r.t<ts(day+1)))
        totals[day]={k:np.zeros(4) for k in names}
        begin=time.time()
        for st in range(0,len(rows),args.chunk):
            pick=rows[st:st+args.chunk];pos=r.positions(pick);sz=np.diff(r.off)[pick]
            xx,ff=data(r,pick);yy=r.labels[pos]
            cc=zscore(content[pos],sz)
            for name in names:
                if name=="content": pred=cc
                else:
                    pred=models[name].predict(ff if "future" in name else xx,num_threads=4)
                    if name.startswith("content_"):pred+=cc
                totals[day][name]+=row_metrics(yy,sz,pred).sum(0)
            done=st+len(pick)
            record(prog,dict(stage="heldout_evaluation",day=day,done=done,total=len(rows),
                            eta_day_sec=(len(rows)-done)*(time.time()-begin)/done))
        results[str(day)]={k:summarized(v,len(rows)) for k,v in totals[day].items()}
    n=sum(results[d]["content"]["impressions"] for d in ("13","14"))
    all_metrics={k:summarized(totals[13][k]+totals[14][k],n) for k in names}
    auc={k:v["AUC"] for k,v in all_metrics.items()}
    out=dict(fit_date="Nov 12",fit_groups=len(chosen),test_dates="Nov 13–14",unlabeled_window="Nov 9–14 only",
             content="BGE-NRMS trained on Nov 9–11 only, fixed two epochs",trees=args.rounds,seed=1119,
             all=all_metrics,days=results,delta_future_vs_prefix=auc["prefix_future"]-auc["prefix"],
             delta_prefix_added_to_content=auc["content_prefix"]-auc["content"],
             delta_future_added_to_content=auc["content_prefix_future"]-auc["content"],
             delta_future_vs_prefix_with_content=auc["content_prefix_future"]-auc["content_prefix"])
    record(HERE/"results.json",out);record(prog,dict(stage="DONE",path=str(HERE/"results.json")))
    print(out,flush=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--fit-groups",type=int,default=60000)
    p.add_argument("--rounds",type=int,default=80);p.add_argument("--chunk",type=int,default=8000)
    main(p.parse_args())

