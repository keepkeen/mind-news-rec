"""Cache BERT layer 10 once, then train two upper layers on clicked vs unclicked candidates."""
import fcntl
import json
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from transformers import AutoModel,AutoTokenizer
from jobio import CACHE,DATA,EXPORT,status,wait_for,write_json

MODEL="BAAI/bge-base-en-v1.5"
REV="a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"
LENGTH=48
class Upper(nn.Module):
    def __init__(self,layers,hidden,masks,host=False):
        super().__init__()
        self.layers=nn.ModuleList(layers)
        self.host=host
        if host:self.host_hidden=hidden
        else:self.register_buffer("hidden",hidden,persistent=False)
        self.register_buffer("masks",masks,persistent=False)
    def forward(self,ids):
        h=self.host_hidden[ids.cpu().long()].to(device=ids.device,dtype=torch.bfloat16) if self.host else self.hidden[ids].to(torch.bfloat16)
        mask=(~self.masks[ids])[:,None,None,:].to(h.dtype)*-10000.
        for layer in self.layers:
            h=layer(h,attention_mask=mask)
            if isinstance(h,tuple): h=h[0]
        return F.normalize(h[:,0].float(),dim=-1)

def load_upper(host=False):
    with (CACHE/"bert10.cache.lock").open("a") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        return _load_upper(host)

def _load_upper(host):
    wait_for(DATA/"shared.ready")
    tok=AutoTokenizer.from_pretrained(MODEL,revision=REV)
    bert=AutoModel.from_pretrained(MODEL,revision=REV,attn_implementation="eager").eval().cuda()
    with (DATA/"news.jsonl").open() as f: rows=[json.loads(l) for l in f]
    n=len(rows)
    hpath=CACHE/"bert10.hidden.npy";mpath=CACHE/"bert10.mask.npy";rpath=CACHE/"bert10.cls.npy"
    ready=CACHE/"bert10.ready"
    if not ready.exists():
        hs=np.lib.format.open_memmap(hpath,mode="w+",dtype="float16",shape=(n,LENGTH,768))
        masks=np.lib.format.open_memmap(mpath,mode="w+",dtype=bool,shape=(n,LENGTH))
        reps=np.lib.format.open_memmap(rpath,mode="w+",dtype="float16",shape=(n,768))
        begin=time.time()
        with torch.inference_mode(),torch.autocast("cuda",dtype=torch.bfloat16):
            for st in range(0,n,192):
                sub=rows[st:st+192]
                x=tok([v[1]+". "+v[2] for v in sub],padding="max_length",truncation=True,max_length=LENGTH,return_tensors="pt").to("cuda")
                out=bert(**x,output_hidden_states=True)
                hs[st:st+len(sub)]=out.hidden_states[-3].to(torch.float16).cpu().numpy()
                masks[st:st+len(sub)]=x.attention_mask.bool().cpu().numpy()
                reps[st:st+len(sub)]=F.normalize(out.last_hidden_state[:,0].float(),dim=-1).cpu().numpy()
                if st==0 or st%(192*100)==0:
                    status("upper","caching_lower10",done=st+len(sub),total=n,sec=time.time()-begin)
        reps[0]=0
        for x in (hs,masks,reps):x.flush()
        del hs,masks,reps
        ready.touch()
    def tensor(p):return torch.from_numpy(np.load(p,mmap_mode="r")).cuda()
    hidden=torch.from_numpy(np.load(hpath,mmap_mode="r")) if host else tensor(hpath)
    upper=Upper(list(bert.encoder.layer[-2:]),hidden,tensor(mpath),host)
    del bert
    raw=tensor(rpath)
    upper.eval()
    with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
        ids=torch.arange(1,120,device="cuda")
        q=upper(ids);cos=(q*F.normalize(raw[ids].float(),dim=-1)).sum(-1).min().item()
        if cos<.995:raise ValueError(f"Layer-cache roundtrip cosine {cos}")
    write_json(EXPORT/"upper.model.json",dict(hf_id=MODEL,revision=REV,token_length=LENGTH,
               frozen_lower_layers=10,trainable_upper_layers=2,cache_fidelity_min_cosine=cos,
               input="title + abstract",gradient_path="candidate upper PLM and user head; periodically refreshed history text"))
    return raw,upper

