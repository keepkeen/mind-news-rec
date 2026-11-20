"""Generic article encoders and deterministic, length-bucketed article batches."""
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer,AutoModel
from jobio import DATA,CACHE

SPECS={
    "fullbge":("BAAI/bge-base-en-v1.5","a5beb1e3e68b9ab74eb54cfd186867f64f240e1a"),
    "e5":("intfloat/e5-base-v2","f52bf8ec8c7124536f0efb74aca902b2995e5bcd"),
    "qlora":("Qwen/Qwen3-Embedding-0.6B","97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"),
    "asym4":("Qwen/Qwen3-Embedding-4B","5cf2132abc99cad020ac570b19d031efec650f2b")
}
SPECS["fullbge_s2"]=SPECS["fullbge"]

class LoRALinear(nn.Module):
    """Rank-16 additive update around an unchanged dense projection."""
    def __init__(self,base,rank=16,alpha=32):
        super().__init__()
        self.base=base
        self.lora_A=nn.Linear(base.in_features,rank,bias=False,device=base.weight.device,dtype=torch.float32)
        self.lora_B=nn.Linear(rank,base.out_features,bias=False,device=base.weight.device,dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_A.weight,a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        self.scale=alpha/rank
    @property
    def weight(self):
        return self.base.weight
    def forward(self,x):
        return self.base(x)+self.scale*self.lora_B(self.lora_A(x))

class ArticleEncoder(nn.Module):
    def __init__(self,kind,length):
        super().__init__()
        self.kind=kind
        hf,rev=SPECS[kind]
        self.lora=kind in ("qlora","asym4")
        tok=AutoTokenizer.from_pretrained(hf,revision=rev,padding_side="right")
        if tok.pad_token_id is None:tok.pad_token=tok.eos_token
        token_kind="fullbge" if kind=="fullbge_s2" else kind
        path=CACHE/f"{token_kind}_ids_{length}.npz"
        if not path.exists():
            ids=[];sizes=[]
            with (DATA/"news.jsonl").open() as f: rows=[json.loads(l) for l in f]
            for start in range(0,len(rows),4096):
                contexts=[r[1]+". "+r[2] if r[2] else r[1] for r in rows[start:start+4096]]
                if kind=="e5":contexts=["query: "+s for s in contexts]
                x=tok(contexts,padding="max_length",truncation=True,max_length=length,return_tensors="np")
                ids.append(x.input_ids.astype(np.int32));sizes.append(x.attention_mask.sum(1).astype(np.int16))
            tmp=path.with_suffix(".tmp")
            with tmp.open("wb") as f:np.savez(f,ids=np.concatenate(ids),length=np.concatenate(sizes))
            tmp.replace(path)
        with np.load(path) as x:
            self.lengths=x["length"]; self.register_buffer("tokens",torch.tensor(x["ids"],device="cuda"),persistent=False)
        self.register_buffer("lens",torch.tensor(self.lengths.astype(np.int64),device="cuda"),persistent=False)
        self.register_buffer("positions",torch.arange(length,device="cuda"),persistent=False)
        self.net=AutoModel.from_pretrained(hf,revision=rev,attn_implementation="sdpa",
                                          dtype=torch.bfloat16 if self.lora else torch.float32).cuda()
        self.width=self.net.config.hidden_size
        if self.lora:
            self.net.requires_grad_(False)
            targets=[(name,m) for name,m in self.net.named_modules() if isinstance(m,nn.Linear)
                     and name.rsplit(".",1)[-1] in ("q_proj","k_proj","v_proj","o_proj")]
            if len(targets)!=self.net.config.num_hidden_layers*4:raise ValueError("Unexpected Qwen projection layout")
            for name,original in targets:
                parent,attribute=name.rsplit(".",1)
                setattr(self.net.get_submodule(parent),attribute,LoRALinear(original))
        else:
            # CLS is read before the optional BERT pooler; every transformer block remains trainable.
            pooler=getattr(self.net,"pooler",None)
            if pooler is not None:pooler.requires_grad_(False)
        self.eval()

    def forward(self,ids,width):
        tokens=self.tokens[ids,:width];mask=self.positions[:width][None,:]<self.lens[ids,None]
        kwargs={"use_cache":False} if self.lora else {}
        h=self.net(input_ids=tokens,attention_mask=mask,**kwargs).last_hidden_state
        if self.kind=="e5":
            v=h.float().masked_fill(~mask.unsqueeze(-1),0.).sum(1)/mask.sum(1,keepdim=True)
        elif self.lora:v=h[torch.arange(len(ids),device="cuda"),self.lens[ids]-1]
        else:v=h[:,0]
        return F.normalize(v.float(),dim=-1)

    def plan(self,ids,micro):
        cpu=ids.detach().cpu().numpy()
        ix=np.flatnonzero(cpu!=0)
        ix=ix[np.argsort(self.lengths[cpu[ix]],kind="stable")]
        pieces=[]
        for st in range(0,len(ix),micro):
            sub=ix[st:st+micro];n=int(self.lengths[cpu[sub[-1]]])
            width=min(self.tokens.shape[1],8*((n+7)//8))
            pieces.append((torch.tensor(sub,device="cuda"),width))
        return pieces

    @torch.no_grad()
    def cache(self,ids,pieces):
        raw=torch.zeros(len(ids),self.width,device="cuda",dtype=torch.float32)
        with torch.autocast("cuda",dtype=torch.bfloat16):
            for ix,width in pieces:raw[ix]=self(ids[ix],width)
        return raw

    @torch.no_grad()
    def all_articles(self,out,micro):
        for st in range(0,len(out),4096):
            ids=torch.arange(st,min(st+4096,len(out)),device="cuda")
            out[st:st+len(ids)]=self.cache(ids,self.plan(ids,micro)).to(out.dtype)

    def save_trainable(self):
        return {n:p.detach().cpu() for n,p in self.named_parameters() if p.requires_grad}

