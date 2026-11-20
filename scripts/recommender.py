"""NRMS / candidate-weighted multi-interest models with split-level inference caching."""
import argparse
from pathlib import Path
import math
import os
import time

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from jobio import DATA,CACHE,RUN,EXPORT,export_file,run_job,status,wait_for,write_json,refit_receipt
from ranking import arr,metrics,pack

def gpu(x):
    return torch.from_numpy(np.asarray(x).copy()).cuda()

class Split:
    def __init__(self,name):
        wait_for(DATA/(name+".ready"))
        self.name=name
        self.off=arr(DATA,name,"offset")
        self.c=arr(DATA,name,"candidate")
        self._c_gpu=None
        self.users=arr(DATA,name,"user")
        self.users_gpu=gpu(self.users)
        ho=arr(DATA,name,"hist_offset")
        hist=arr(DATA,name,"hist")
        padded=np.zeros((len(ho)-1,50),np.int32)
        for st in range(0,len(padded),16384):
            end=min(st+16384,len(padded))
            ix=ho[st+1:end+1,None]+np.arange(-50,0)
            valid=ix>=ho[st:end,None]
            padded[st:end]=np.where(valid,hist[np.maximum(ix,0)],0)
        self.history=gpu(padded)

    @property
    def c_gpu(self):
        if self._c_gpu is None:self._c_gpu=gpu(self.c)
        return self._c_gpu

class TrainingPairs:
    def __init__(self,splits):
        pos, groups, neg, neg_len, weights, user, histories=[],[],[],[],[],[],[]
        group_base=user_base=0
        for s in splits:
            y=arr(DATA,s.name,"label")
            npos=np.add.reduceat(y,s.off[:-1],dtype=np.int32)
            nneg=np.diff(s.off)-npos
            assert np.all(nneg>0) and np.all(npos>0)
            pos.append(s.c[y==1]);neg.append(s.c[y==0])
            groups.append(np.repeat(np.arange(len(npos),dtype=np.int32)+group_base,npos))
            neg_len.append(nneg);weights.append(np.float32(1)/npos.astype(np.float32))
            user.append(np.asarray(s.users)+user_base);histories.append(s.history)
            group_base+=len(npos);user_base+=len(s.history)
        counts=np.concatenate(neg_len)
        self.neg_start=gpu(np.cumsum(counts,dtype=np.int32)-counts)
        self.neg_count=gpu(counts)
        self.neg=gpu(np.concatenate(neg));self.pos=gpu(np.concatenate(pos))
        self.groups=gpu(np.concatenate(groups));self.w=gpu(np.concatenate(weights))
        self.u=gpu(np.concatenate(user));self.hist=torch.cat(histories) if len(histories)>1 else histories[0]
    def batch(self,index):
        g=self.groups[index]
        q=(torch.rand(len(g),4,device="cuda")*self.neg_count[g,None]).int()+self.neg_start[g,None]
        ids=torch.cat((self.pos[index,None],self.neg[q]),dim=1)
        return self.hist[self.u[g]],ids,self.w[g]

class Model(nn.Module):
    def __init__(self,raw,cats,subs,heads,upper=None,dim=160):
        super().__init__()
        self.register_buffer("raw",raw,persistent=False)
        self.register_buffer("cats",cats,persistent=False)
        self.register_buffer("subs",subs,persistent=False)
        self.upper=upper
        self.dim=dim
        self.text=nn.Sequential(nn.LayerNorm(raw.shape[1]),nn.Linear(raw.shape[1],dim))
        self.category=nn.Embedding(int(cats.max())+1,dim,padding_idx=0)
        self.subcategory=nn.Embedding(int(subs.max())+1,dim,padding_idx=0)
        nn.init.normal_(self.category.weight,std=.06);nn.init.normal_(self.subcategory.weight,std=.06)
        self.news_norm=nn.LayerNorm(dim)
        self.bias=nn.Linear(dim,1)
        self.position=nn.Parameter(torch.randn(50,dim)*.03)
        self.qkv=nn.Linear(dim,3*dim)
        self.context=nn.Linear(dim,dim)
        self.history_norm=nn.LayerNorm(dim)
        self.pool=nn.Sequential(nn.Linear(dim,80),nn.Tanh(),nn.Linear(80,heads))
        self.user_out=nn.Linear(dim,dim)
        self.drop=nn.Dropout(.12)
    def news(self,ids,live=False):
        if self.upper is not None and live:
            uniq,inv=torch.unique(ids,return_inverse=True)
            x=self.upper(uniq)[inv.reshape(ids.shape)]
        else:
            x=self.raw[ids]
        return self.project(x,ids)
    def project(self,x,ids):
        v=self.text(x.float())+self.category(self.cats[ids])+self.subcategory(self.subs[ids])
        v=self.news_norm(F.gelu(v))
        return v,self.bias(v).squeeze(-1)
    def user(self,h,ids):
        mask=ids!=0
        valid=mask.clone()
        valid[:,0] |= ~mask.any(1)
        h=self.drop(h+self.position)
        q,k,v=self.qkv(h).reshape(len(h),50,3,4,self.dim//4).permute(2,0,3,1,4).unbind(0)
        ctx=F.scaled_dot_product_attention(q,k,v,attn_mask=valid[:,None,None,:])
        h=self.history_norm(h+self.context(ctx.transpose(1,2).reshape_as(h)))
        weights=self.pool(h).masked_fill(~valid[:,:,None],-1e4).softmax(1)
        u=self.user_out(weights.transpose(1,2)@h)
        return u*mask.any(1)[:,None,None]
    def combine(self,dots,bias):
        dots=dots.float()/math.sqrt(self.dim)
        if dots.shape[-1]==1: val=dots.squeeze(-1)
        else: val=(dots.softmax(-1)*dots).sum(-1)
        return val+bias.float()
    def forward(self,hid,cid):
        hv,_=self.news(hid)
        u=self.user(hv,hid)
        v,b=self.news(cid,live=self.upper is not None)
        return self.combine(v@u.transpose(1,2),b)
    @torch.no_grad()
    def refresh(self):
        if self.upper is None: return
        state=self.upper.training;self.upper.eval()
        for st in range(1,len(self.raw),384):
            en=min(st+384,len(self.raw))
            ids=torch.arange(st,en,device="cuda")
            self.raw[st:en]=self.upper(ids).to(self.raw.dtype)
        self.upper.train(state)

@torch.no_grad()
def score(model,s,tag):
    model.eval()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        nv=[];nb=[];historical=[]
        for st in range(0,len(model.raw),1024):
            ids=torch.arange(st,min(st+1024,len(model.raw)),device="cuda")
            v,b=model.news(ids)
            if hasattr(model,"candidate"):
                historical.append(v.to(torch.bfloat16))
                v,b=model.candidate(model.candidate_raw[ids],ids)
            nv.append(v.to(torch.bfloat16));nb.append(b.float())
        v=torch.cat(nv);bias=torch.cat(nb)
        hist_v=torch.cat(historical) if historical else v
        us=[]
        for st in range(0,len(s.history),768):
            h=s.history[st:st+768];us.append(model.user(hist_v[h],h).to(torch.bfloat16))
        users=torch.cat(us)
        path=CACHE/(tag+"_"+s.name+".npy")
        output=np.lib.format.open_memmap(path,mode="w+",dtype=np.float32,shape=(len(s.c),))
        for i in range(0,len(s.off)-1,1024):
            en=min(i+1024,len(s.off)-1);lo=int(s.off[i]);hi=int(s.off[en])
            repeats=gpu(np.diff(s.off[i:en+1])).long()
            uid=torch.repeat_interleave(s.users_gpu[i:en],repeats,output_size=hi-lo)
            cid=s.c_gpu[lo:hi]
            # FP32 accumulation preserves small ordering differences at score export.
            dots=(users[uid].float()*v[cid,None,:].float()).sum(-1)
            output[lo:hi]=model.combine(dots,bias[cid]).cpu().numpy()
        output.flush()
    return path,output

def train(a):
    torch.manual_seed(a.seed);np.random.seed(a.seed)
    torch.set_float32_matmul_precision("high")
    series=a.series or a.kind
    job=series+("_shadow" if a.shadow else "_refit" if a.refit else "")
    status(job,"waiting_inputs")
    upper=None
    if a.kind=="upper":
        from upper import load_upper
        raw,upper=load_upper(a.host_upper_cache)
    else:
        wait_for(CACHE/(a.kind+".npy"))
        raw=gpu(np.load(CACHE/(a.kind+".npy"),mmap_mode="r"))
    splits=[Split("train")]
    dev=Split("dev")
    if a.refit:splits.append(dev)
    data=TrainingPairs(splits)
    use=None
    if a.shadow:
        # Click-supervised content for the train-week measurement: labels through Nov 11 only.
        end=int(np.datetime64("2019-11-12T00:00:00").astype("int64"))
        eligible=gpu(np.asarray(arr(DATA,"train","time"))<end)
        use=torch.where(eligible[data.groups])[0]
    model=Model(raw,gpu(np.load(DATA/"cat.npy").astype("int32")),
                gpu(np.load(DATA/"subcat.npy").astype("int32")),4 if a.kind=="qwen" else 1,upper).cuda()
    pars=[p for n,p in model.named_parameters() if not n.startswith("upper.")]
    param_groups=[{"params":pars,"lr":a.lr}]
    if upper is not None:param_groups.append({"params":list(upper.parameters()),"lr":1.5e-5})
    opt=torch.optim.AdamW(param_groups,weight_decay=.01)
    ntrain=len(use) if use is not None else len(data.pos)
    steps=math.ceil(ntrain/a.batch)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min(1,(s+1)/min(180,steps*.05))*(.35+.65*(1+math.cos(math.pi*s/(steps*a.epochs)))/2))
    torch.cuda.reset_peak_memory_stats()
    # Representative forward and backward with a full batch, cache path and all real label shapes.
    model.train()
    ix=torch.linspace(0,ntrain-1,a.batch,device="cuda").long()
    if use is not None: ix=use[ix]
    hid,cid,w=data.batch(ix)
    with torch.autocast("cuda",dtype=torch.bfloat16):
        logits=model(hid,cid);loss=(F.cross_entropy(logits,torch.zeros(len(w),device="cuda",dtype=torch.long),reduction="none")*w).mean()/w.mean()
    loss.backward()
    if not torch.isfinite(loss): raise ValueError("Nonfinite smoke loss")
    opt.zero_grad(set_to_none=True)
    model.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        direct=model(hid[:8],cid[:8]);hv,_=model.news(hid[:8])
        target,bias=model.news(cid[:8]);user=model.user(hv,hid[:8])
        cached=model.combine((user[:,None].float()*target[:,:,None].float()).sum(-1),bias)
        diff=float((direct-cached).abs().max().cpu())
        if diff>.14:raise ValueError(f"Cached score fidelity {diff}")
    status(job,"smoke_ok",peak_gib=torch.cuda.max_memory_allocated()/2**30,cache_max_error=diff,
           impressions=int(eligible.sum()) if a.shadow else sum(len(s.off)-1 for s in splits),positive_pairs=ntrain,steps_per_epoch=steps)
    best=-1;best_epoch=0;test=None
    for ep in range(1,a.epochs+1):
        model.train(); begin=time.time();order=torch.randperm(ntrain,device="cuda")
        if use is not None: order=use[order]
        for st in range(0,len(order),a.batch):
            it=st//a.batch
            hid,cid,w=data.batch(order[st:st+a.batch])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=model(hid,cid)
                losses=F.cross_entropy(logits,torch.zeros(len(w),device="cuda",dtype=torch.long),reduction="none")
                loss=(losses*w).sum()/w.sum()
            opt.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step();sched.step()
            if upper is not None and (it+1)%a.refresh==0:
                with torch.autocast("cuda",dtype=torch.bfloat16):model.refresh()
            if it in (0,20,100) or (it+1)%300==0:
                elapsed=time.time()-begin
                status(job,"training",epoch=ep,epochs=a.epochs,step=it+1,steps=steps,loss=loss.item(),
                       sec_per_step=elapsed/(it+1),epoch_eta_sec=(steps-it-1)*elapsed/(it+1))
        with torch.autocast("cuda",dtype=torch.bfloat16): model.refresh()
        tag=f"{job}_e{ep}"
        if a.shadow:
            if ep==a.epochs:
                path,_=score(model,splits[0],tag);export_file(path)
                write_json(EXPORT/(series+"_shadow_meta.json"),dict(epoch=ep,score=path.name,fit_before="2019-11-12",seed=a.seed))
                status(job,"DONE",fit_before="2019-11-12",epoch=ep,train_scores=path.name)
            continue
        if a.refit:
            improved=True;metrics_ep={}
        else:
            dev_path,dev_scores=score(model,dev,tag)
            metrics_ep=metrics(DATA,"dev",dev_scores)
            write_json(EXPORT/(tag+"_metrics.json"),metrics_ep)
            auc=metrics_ep["user_half_fit" if a.series else "all"]["AUC"];improved=auc>best
            if improved:
                best=auc;export_file(dev_path)
        if improved:
            best_epoch=ep
            ck=CACHE/(tag+".pt")
            torch.save(dict(model=model.state_dict(),epoch=ep,seed=a.seed,kind=a.kind,refit=a.refit),ck);export_file(ck)
            # A valid first-epoch backup is published before training the second epoch.
            # Other models also keep test logits for direct score fusion.
            if a.refit or not a.series:
                if test is None:test=Split("test")
                status(job,"test_scoring",epoch=ep,dev=metrics_ep.get("all"))
                path,test_scores=score(model,test,tag)
                if a.kind=="bge" and not a.refit:
                    zip_path=CACHE/(tag+".zip");pack(DATA,"test",test_scores,zip_path);export_file(zip_path)
                export_file(path)
                if a.refit:refit_receipt(series,ep,a.seed,ck,path,ntrain,a.batch)
        if not a.refit:
            write_json(EXPORT/(job+"_best.json"),{"job":job,"best_epoch":best_epoch,"auc":best,"seed":a.seed})
    if not a.shadow: status(job,"DONE",best_epoch=None if a.refit else best_epoch,epochs_saved=a.epochs if a.refit else None,AUC=None if a.refit else best)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=["bge","qwen","upper"]);p.add_argument("epochs",type=int)
    p.add_argument("batch",type=int);p.add_argument("lr",type=float);p.add_argument("seed",type=int)
    mode=p.add_mutually_exclusive_group();mode.add_argument("--refit",action="store_true");mode.add_argument("--shadow",action="store_true")
    p.add_argument("--refresh",type=int,default=500);p.add_argument("--series",choices=["upper_s2","upper_s3"])
    p.add_argument("--host-upper-cache",action="store_true")
    a=p.parse_args();job=(a.series or a.kind)+("_shadow" if a.shadow else "_refit" if a.refit else "")
    run_job(job,lambda:train(a))

