"""Frozen generic encoders. No recommendation/prediction weights enter this pipeline."""
import argparse
import json
import time
import os
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer,AutoModel
from jobio import DATA,CACHE,EXPORT,run_job,status,wait_for,write_json,export_file

def main(a):
    job="embed_"+a.kind
    status(job,"loading_model",model=a.model,revision=a.revision)
    tok=AutoTokenizer.from_pretrained(a.model,revision=a.revision)
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    is_qwen=a.kind=="qwen"
    model=AutoModel.from_pretrained(a.model,revision=a.revision,torch_dtype=torch.bfloat16).eval().cuda()
    wait_for(DATA/"shared.ready")
    with (DATA/"news.jsonl").open() as f: items=[json.loads(l) for l in f]
    text=[r[1]+(". "+r[2] if r[2] else "") for r in items]
    order=sorted(range(1,len(text)),key=lambda i:len(text[i]))
    width=model.config.hidden_size
    dst=CACHE/(a.kind+".npy")
    out=np.lib.format.open_memmap(str(dst)+".partial",mode="w+",dtype="float16",shape=(len(text),width))
    out[0]=0
    def batch(ids):
        x=tok([text[i] for i in ids],padding=True,truncation=True,max_length=a.length,return_tensors="pt").to("cuda")
        kwargs={"use_cache":False} if is_qwen else {}
        h=model(**x,**kwargs).last_hidden_state
        v=h[torch.arange(len(ids),device="cuda"),x.attention_mask.sum(1)-1] if is_qwen else h[:,0]
        return F.normalize(v.float(),dim=-1).cpu().numpy()
    with torch.inference_mode():
        # Include both the longest titles/abstracts and different lengths in the smoke.
        ids=order[-a.batch:]
        x=batch(ids);y=batch(ids[:5])
        cos=np.sum(x[:5]*y,axis=1)
        if not np.isfinite(x).all() or np.min(cos)<0.995: raise ValueError("Batch fidelity "+str(cos))
        status(job,"smoke_ok",batch=len(ids),max_tokens=a.length,min_cosine=float(cos.min()),
               peak_gib=torch.cuda.max_memory_allocated()/2**30,gpu=torch.cuda.get_device_name())
        begin=time.time()
        for st in range(0,len(order),a.batch):
            ids=order[st:st+a.batch];out[ids]=batch(ids)
            done=st+len(ids)
            if st==0 or (st//a.batch)%80==0 or done==len(order):
                sec=time.time()-begin
                status(job,"encoding",done=done,total=len(order),items_per_s=done/sec,
                       eta_sec=(len(order)-done)*sec/done)
    out.flush();del out;os.replace(str(dst)+".partial",dst)
    write_json(EXPORT/(a.kind+".model.json"),dict(hf_id=a.model,revision=a.revision,encoder="frozen",
             pooling="last non-padding" if is_qwen else "CLS",l2_normalized=True,input="title + abstract",
             maximum_tokens=a.length,articles=len(order),width=width,device=torch.cuda.get_device_name()))
    # Baseline vectors also power the local train-week content control.
    if not is_qwen: export_file(dst)
    status(job,"DONE",seconds=time.time()-begin,items=len(order))

if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("kind",choices=["bge","qwen"]);p.add_argument("model");p.add_argument("revision")
    p.add_argument("batch",type=int);p.add_argument("length",type=int)
    a=p.parse_args();run_job("embed_"+a.kind,lambda:main(a))

