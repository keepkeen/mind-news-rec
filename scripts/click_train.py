"""Click-supervised PLM tuning with exact gradient replay across all distinct articles per batch."""
import argparse
from datetime import datetime,timezone
import json
import math
import os
import statistics
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from click_text import ArticleEncoder,SPECS
from jobio import DATA,CACHE,EXPORT,RUN,write_json,export_file,run_job,status,refit_receipt
from recommender import Split,TrainingPairs,Model,gpu,score
from ranking import arr,metrics

def draw(data,index):
    groups=data.groups[index];n=data.neg_count[groups];offset=data.neg_start[groups]
    draws=[]
    for j in range(4):
        slot=(torch.rand(len(n),device="cuda")*(n-j).clamp_min(1)).int()
        if draws:
            prev=torch.stack(draws,dim=1).sort(1).values
            for k in range(j):slot=slot+(slot>=prev[:,k]).int()
            slot=torch.where(n>j,slot,(torch.rand(len(n),device="cuda")*n).int())
        draws.append(slot)
    neg=data.neg[offset[:,None]+torch.stack(draws,dim=1)]
    return data.hist[data.u[groups]],torch.cat([data.pos[index,None],neg],1),data.w[groups]

def cached_step(encoder,head,data,index,opt,text_micro,head_micro):
    hid,cid,w=draw(data,index)
    count=hid.numel()
    asym=hasattr(head,"candidate")
    if asym:
        uid,ci=torch.unique(cid,return_inverse=True)
    else:
        uid,inverse=torch.unique(torch.cat([hid.flatten(),cid.flatten()]),return_inverse=True)
        hi=inverse[:count].reshape_as(hid);ci=inverse[count:].reshape_as(cid)
    chunks=encoder.plan(uid,text_micro)
    raw=encoder.cache(uid,chunks).requires_grad_(True)
    opt.zero_grad(set_to_none=True)
    norm=w.sum()
    total=torch.zeros((),device="cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        for st in range(0,len(w),head_micro):
            sl=slice(st,st+head_micro)
            h,_=head.news(hid[sl]) if asym else head.project(raw[hi[sl]],hid[sl])
            u=head.user(h,hid[sl])
            v,b=(head.candidate if asym else head.project)(raw[ci[sl]],cid[sl])
            logits=head.combine(v@u.transpose(1,2),b)
            loss=((-F.log_softmax(logits,1)[:,0])*w[sl]).sum()/norm
            loss.backward();total+=loss.detach()
        # One upstream gradient per distinct news, including every historical and candidate occurrence.
        for ix,width in chunks:
            result=encoder(uid[ix],width)
            torch.autograd.backward(result,raw.grad[ix])
    params=[p for group in opt.param_groups for p in group["params"]]
    grad=nn.utils.clip_grad_norm_(params,2.)
    opt.step()
    return total,grad,len(uid)

def replay_parity(encoder,head,data,eligible):
    """Compare the complete sampled ranking loss and encoder/head gradients with direct autograd."""
    head.eval()
    h,c,w=draw(data,eligible[:3])
    # A small, nonempty slice makes the direct graph fit both 40GB profiles.
    h=h[:,-8:];h=F.pad(h,(42,0))
    asym=hasattr(head,"candidate")
    if asym:unique,ic=torch.unique(c,return_inverse=True)
    else:
        unique,inv=torch.unique(torch.cat([h.flatten(),c.flatten()]),return_inverse=True)
        ih=inv[:h.numel()].reshape_as(h);ic=inv[h.numel():].reshape_as(c)
    real=torch.where(unique!=0)[0]
    width=min(encoder.tokens.shape[1],8*((int(encoder.lengths[unique[real].cpu().numpy()].max())+7)//8))
    named=[(n,p) for n,p in encoder.named_parameters() if p.requires_grad and (not encoder.lora or "lora_B" in n)]
    first=next((p for n,p in named if "layer.0." in n or "layers.0." in n),named[0][1])
    last=named[-1][1]
    def loss(raw):
        hv,_=head.news(h) if asym else head.project(raw[ih],h)
        u=head.user(hv,h);v,b=(head.candidate if asym else head.project)(raw[ic],c)
        l=head.combine(v@u.transpose(1,2),b)
        return ((-F.log_softmax(l,dim=1)[:,0])*w).sum()/w.sum()
    with torch.autocast("cuda",dtype=torch.bfloat16):
        direct=encoder(unique[real],width)
        full=torch.zeros(len(unique),encoder.width,device="cuda").index_copy(0,real,direct)
        ld=loss(full)
        head_params=[head.bias.weight,head.text[1].weight] if asym else [head.bias.weight]
        direct_grads=torch.autograd.grad(ld,[first,last,*head_params])
        raw=full.detach().requires_grad_(True)
        lc=loss(raw);up,*head_grads=torch.autograd.grad(lc,[raw,*head_params])
        replay=encoder(unique[real],width)
        back=torch.autograd.grad(replay,[first,last],grad_outputs=up[real])
    rel=[]
    for a,b in zip(direct_grads,(*back,*head_grads)):
        norm=a.float().norm().item()
        if not math.isfinite(norm) or norm==0:raise ValueError("Missing gradient in cache parity")
        rel.append(((a.float()-b.float()).norm()/norm).item())
    if max(rel)>.025 or abs(float((ld-lc).detach()))>1e-4:raise ValueError(f"Gradient replay mismatch {rel}")
    head.train()
    return dict(relative_grad_errors=rel,loss_difference=abs(float(ld.detach()-lc.detach())))

def warm_head(head,data,eligible,epochs,seed):
    opt=torch.optim.AdamW(head.parameters(),lr=.0007,weight_decay=.01)
    head.train()
    steps=math.ceil(len(eligible)/768)*epochs
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda x:min((x+1)/80,1.)*(.4+.6*(1+math.cos(math.pi*x/steps))/2))
    for ep in range(epochs):
        order=eligible[torch.randperm(len(eligible),device="cuda")]
        for st in range(0,len(order),768):
            h,c,w=draw(data,order[st:st+768])
            with torch.autocast("cuda",dtype=torch.bfloat16):
                logits=head(h,c);loss=((-F.log_softmax(logits,dim=1)[:,0])*w).sum()/w.sum()
            opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(head.parameters(),2.);opt.step();sch.step()

def run(a):
    kind=a.kind;job=kind+("_shadow" if a.shadow else "_refit" if a.refit else "")
    torch.manual_seed(a.seed);np.random.seed(a.seed);torch.set_float32_matmul_precision("high")
    status(job,"loading_generic_encoder",model=SPECS[kind],shadow=a.shadow)
    tr=Split("train")
    splits=[tr,Split("dev")] if a.refit else [tr]
    data=TrainingPairs(splits)
    enc=ArticleEncoder(kind,a.length)
    raw=torch.zeros((len(enc.lengths),enc.width),device="cuda",dtype=torch.float16)
    cats=gpu(np.load(DATA/"cat.npy").astype(np.int32));subs=gpu(np.load(DATA/"subcat.npy").astype(np.int32))
    if kind=="asym4":
        from asym_head import AsymmetricHead
        history=gpu(np.load(CACHE/"qwen.npy",mmap_mode="r"))
        if tuple(history.shape)!=(len(enc.lengths),4096):raise ValueError("Unexpected 8B history cache")
        head=AsymmetricHead(history,raw,cats,subs).cuda()
    else:head=Model(raw,cats,subs,4).cuda()
    eligible=torch.arange(len(data.pos),device="cuda")
    if a.shadow:
        cutoff=int(np.datetime64("2019-11-12T00:00:00").astype("int64"))
        eligible=eligible[gpu(np.asarray(arr(DATA,"train","time"))<cutoff)[data.groups]]
    # Shadow training uses the selected epoch count and only clicks before the tree-fit window.
    full_meta=EXPORT/(kind+"_best.json")
    epochs=int(json.loads(full_meta.read_text())["best_epoch"]) if a.shadow else a.epochs
    n=len(eligible);steps=math.ceil(n/a.batch)
    text_params=[p for p in enc.parameters() if p.requires_grad]
    def optim(text_lr,head_lr):
        return torch.optim.AdamW([dict(params=text_params,lr=text_lr),dict(params=head.parameters(),lr=head_lr)],
                                 weight_decay=.01,fused=True)
    parity=replay_parity(enc,head,data,eligible)
    micro=a.micro
    def timed_smoke(micro,examples=None,early=False):
        examples=eligible if examples is None else examples
        opt=optim(0.,0.)
        timings=[];distinct=[]
        repeats=2 if enc.lora and a.batch>=32768 else 3
        for k in range(repeats):
            ix=examples[torch.randperm(len(examples),device="cuda")[:a.batch]]
            torch.cuda.synchronize();start=time.perf_counter()
            loss,grad,unique=cached_step(enc,head,data,ix,opt,micro,384)
            torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if not torch.isfinite(loss) or not torch.isfinite(grad) or grad.item()<=0:raise RuntimeError("Nonfinite/zero smoke gradient")
            timings.append(elapsed);distinct.append(unique)
            status(job,"timing_smoke",done=k+1,total=repeats,seconds_per_step=elapsed,early_shadow_profile=early,
                   logical_positives=a.batch,encoder_micro=micro,distinct_articles=unique)
        opt.zero_grad(set_to_none=True)
        return timings,distinct
    while True:
        try:
            torch.cuda.reset_peak_memory_stats()
            timings,distinct=timed_smoke(micro)
            break
        except torch.OutOfMemoryError:
            if micro<=16:raise
            micro//=2
            status(job,"retry_smoke_same_logical_batch",text_micro=micro)
        enc.zero_grad(set_to_none=True);head.zero_grad(set_to_none=True);torch.cuda.empty_cache()
    sec=max(timings) if len(timings)==2 else statistics.median(timings[1:])
    projected=sec*steps*1.10+180.
    spec=dict(model=SPECS[kind][0],revision=SPECS[kind][1],logical_positives=a.batch,encoder_micro=micro,
        positive_pairs=n,steps_per_epoch=steps,step_seconds=timings,steady_step_seconds=sec,estimated_epoch_sec=projected,
        distinct_articles=distinct,peak_gib=torch.cuda.max_memory_allocated()/2**30,parity=parity,
        epochs_requested=epochs,trainable_text_parameters=sum(p.numel() for p in text_params),
        encoder_mode="all 12 Transformer blocks" if not enc.lora else "rank16 alpha32 LoRA q/k/v/o in every layer",
        history_gradients="all 50 unmasked recent histories and candidates, exact gradient replay",
        loss="all positive clicks; four same-impression negatives, distinct where possible; equal impression weighting",
        text="title and abstract",max_tokens=a.length,candidate_only_bias=True,
        cold_start="content bias for an empty click history",shadow=a.shadow,gpu=torch.cuda.get_device_name())
    if kind=="e5":
        spec.update(pooling="masked mean, FP32 accumulation, L2 normalized",article_prefix="query: ",
                    seed=a.seed,training_splits=[s.name for s in splits],every_epoch_checkpoint=not a.shadow)
        if not a.shadow:
            if not a.refit:
                cutoff=int(np.datetime64("2019-11-12T00:00:00").astype("int64"))
                early=eligible[gpu(np.asarray(arr(DATA,"train","time"))<cutoff)[data.groups]]
                with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                    early_times,_=timed_smoke(micro,early,True)
                spec.update(shadow_step_seconds=early_times,shadow_steps=math.ceil(len(early)/a.batch),
                            shadow_positive_pairs=len(early))
            write_json(EXPORT/(job+"_pilot.json"),spec)
            gate=RUN/"e5_admission.json"
            status(job,"awaiting_paired_admission",epoch_hours=projected/3600)
            limit=time.time()+1500
            while not gate.is_file() and time.time()<limit:time.sleep(4)
            decision=json.loads(gate.read_text()) if gate.is_file() else {"run":False,"reason":"paired pilot unavailable"}
            if not decision["run"]:
                status(job,"SKIPPED",decision=decision);return
    if kind=="asym4":
        spec.update(history_gradients="frozen 8B article vectors, learned history projection; 4B encoder gradients only for candidates",
            history_cache_sha256="f7588663b9d9f4fd44cc740739b48fd43bc320988e9e1b505ac7f4a5fd1739e2",
            training_splits=[s.name for s in splits])
        if not a.shadow:
            write_json(EXPORT/(job+"_pilot.json"),spec)
            gate=RUN/"asym_admission.json"
            status(job,"awaiting_budget_decision",projected_hours=projected/3600)
            limit=time.time()+1200
            while not gate.exists() and time.time()<limit:time.sleep(6)
            decision=json.loads(gate.read_text()) if gate.exists() else {"run":False,"reason":"admission timeout"}
            if projected>6*3600 or not decision["run"]:
                status(job,"SKIPPED",estimated_epoch_sec=projected,decision=decision);return
    if kind=="qlora" and not (a.shadow or a.refit) and projected>6.5*3600:
        spec["decision"]="SKIPPED_SIX_HOUR_GATE";write_json(EXPORT/(job+"_smoke.json"),spec)
        write_json(EXPORT/(job+f"_smoke_b{a.batch}.json"),spec)
        status(job,"SKIPPED",reason="above six-hour epoch gate",**spec);return
    if kind=="qlora" and not (a.shadow or a.refit) and projected>3.5*3600:epochs=1
    if kind=="fullbge" and not (a.shadow or a.refit) and projected>3*3600:epochs=2
    spec["training_splits"]=[s.name for s in splits];spec["every_epoch_checkpoint"]=a.refit or (kind=="e5" and not a.shadow)
    spec["epochs"]=epochs;spec["decision"]="RUN";write_json(EXPORT/(job+"_smoke.json"),spec)
    write_json(EXPORT/(job+f"_smoke_b{a.batch}.json"),spec)
    status(job,"smoke_ok",**spec)
    if not a.resume:
        t=time.time();enc.all_articles(raw,micro)
        status(job,"warm_head",epochs=2,initial_news_cache_seconds=time.time()-t)
        warm_head(head,data,eligible,2,a.seed)
    lr=.00015 if enc.lora else .000022
    opt=optim(lr,.00035);total_steps=steps*epochs
    sch=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:min((s+1)/max(4,int(.06*total_steps)),1.)*
                                           (.15+.85*(1+math.cos(math.pi*s/total_steps))/2))
    best=-1;best_ep=0;dev=None;test=None
    start_epoch=1;resume_step=0;resume_order=None
    if a.resume:
        from pathlib import Path
        from jobio import wait_for
        from mid_checkpoint import restore
        receipt=Path(a.resume).with_suffix(".sha256")
        status(job,"waiting_for_verified_resume",file=Path(a.resume).name)
        if not Path(a.resume).is_file():wait_for(receipt)
        start_epoch,resume_step,resume_order,best,best_ep=restore(a.resume,job,enc,head,opt,sch,spec,a.seed)
        resumed=dict(epoch=start_epoch,completed_step=resume_step,next_step=resume_step+1,
            seed=a.seed,file=Path(a.resume).name,sha256=receipt.read_text().strip() if receipt.exists() else None,
            restored=["encoder","head","optimizer","scheduler","permutation","CPU RNG","CUDA RNG"],utc=time.time())
        write_json(EXPORT/(job+"_resumed.json"),resumed);status(job,"restored",**resumed)
    for ep in range(start_epoch,epochs+1):
        if ep==start_epoch and resume_order is not None:
            order_state=resume_order;g=torch.Generator(device="cuda");g.set_state(order_state)
            order=eligible[torch.randperm(n,device="cuda",generator=g)];first_step=resume_step
        else:
            order_state=torch.cuda.get_rng_state();order=eligible[torch.randperm(n,device="cuda")];first_step=0
        begin=time.time();head.train()
        for st in range(first_step*a.batch,n,a.batch):
            it=st//a.batch+1
            loss,grad,unique=cached_step(enc,head,data,order[st:st+a.batch],opt,micro,384);sch.step()
            if it in (first_step+1,first_step+4) or it%10==0 or it==steps:
                loss_value=loss.item();grad_value=grad.item()
                if not math.isfinite(loss_value+grad_value):raise ValueError("Nonfinite training step")
                elapsed=time.time()-begin
                status(job,"training",epoch=ep,epochs=epochs,step=it,steps=steps,unique_news=unique,loss=loss_value,
                       sec_per_step=elapsed/(it-first_step),epoch_eta_sec=elapsed/(it-first_step)*(steps-it))
            if a.checkpoint_every and (it==1 or it%a.checkpoint_every==0) and it<steps:
                from mid_checkpoint import snapshot
                snapshot(job,ep,it,order_state,enc,head,opt,sch,spec,a.seed,best,best_ep)
        tag=f"{job}_e{ep}";cp=CACHE/(tag+".pt")
        if a.checkpoint_every:
            # Export before evaluation, full-news refresh or scoring can interrupt the epoch boundary.
            torch.save(dict(article_trainable=enc.save_trainable(),head=head.state_dict(),hf=SPECS[kind],
                            spec=spec,epoch=ep,seed=a.seed),cp);export_file(cp)
        if a.shadow and ep!=epochs:continue
        status(job,"refresh_for_evaluation",epoch=ep);enc.all_articles(raw,micro)
        if a.shadow:
            path,_=score(head,tr,tag);export_file(path)
            write_json(EXPORT/(kind+"_shadow_meta.json"),dict(epoch=ep,score=path.name,fit_before="2019-11-12",seed=a.seed))
        elif a.refit:
            if not a.checkpoint_every:
                torch.save(dict(article_trainable=enc.save_trainable(),head=head.state_dict(),
                    hf=SPECS[kind],spec=spec,epoch=ep,seed=a.seed),cp);export_file(cp)
            if kind in ("fullbge_s2","e5"):
                saved=torch.load(cp,map_location="cpu",weights_only=True)
                with torch.no_grad():
                    trainable=dict((k,p) for k,p in enc.named_parameters() if p.requires_grad)
                    if trainable.keys()!=saved["article_trainable"].keys():raise ValueError("Incomplete encoder checkpoint")
                    for name,param in trainable.items():param.copy_(saved["article_trainable"][name])
                    head.load_state_dict(saved["head"],strict=True)
                del saved
                enc.all_articles(raw,micro)
            if test is None:test=Split("test")
            tp,_=score(head,test,tag);export_file(tp)
            refit_receipt(kind,ep,a.seed,cp,tp,n,a.batch,checkpoint_scored=kind in ("fullbge_s2","e5"))
        else:
            if dev is None:dev=Split("dev")
            dp,ds=score(head,dev,tag);res=metrics(DATA,"dev",ds);write_json(EXPORT/(tag+"_metrics.json"),res)
            selection=res["user_half_fit"]["AUC"]
            improved=selection>best
            if improved or kind=="e5":
                export_file(dp)
                if not a.checkpoint_every:
                    torch.save(dict(article_trainable=enc.save_trainable(),head=head.state_dict(),
                        hf=SPECS[kind],spec=spec,epoch=ep,seed=a.seed),cp);export_file(cp)
                if kind not in ("asym4","fullbge_s2","e5"):
                    if test is None:test=Split("test")
                    tp,_=score(head,test,tag);export_file(tp)
            if improved:
                best=selection;best_ep=ep
                write_json(EXPORT/(kind+"_best.json"),dict(job=kind,best_epoch=ep,auc=res["all"]["AUC"],selection_A=best))
    status(job,"DONE",epochs_saved=epochs if a.refit or (kind=="e5" and not a.shadow) else None,best_epoch=None if a.refit else epochs if a.shadow else best_ep,
           selection_A=best if not (a.shadow or a.refit) else None)

if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("kind",choices=SPECS);p.add_argument("batch",type=int);p.add_argument("micro",type=int)
    p.add_argument("length",type=int);p.add_argument("epochs",type=int);p.add_argument("seed",type=int)
    mode=p.add_mutually_exclusive_group();mode.add_argument("--shadow",action="store_true");mode.add_argument("--refit",action="store_true")
    p.add_argument("--checkpoint-every",type=int,default=0);p.add_argument("--resume")
    a=p.parse_args();run_job(a.kind+("_shadow" if a.shadow else "_refit" if a.refit else ""),lambda:run(a))

