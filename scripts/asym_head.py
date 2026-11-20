"""Candidate-only click gradients with a distinct frozen history encoder."""
from torch import nn
import torch.nn.functional as F
from recommender import Model

class AsymmetricHead(Model):
    def __init__(self,history_raw,candidate_raw,cats,subs):
        super().__init__(history_raw,cats,subs,4)
        self.register_buffer("candidate_raw",candidate_raw,persistent=False)
        self.candidate_text=nn.Sequential(nn.LayerNorm(candidate_raw.shape[1]),nn.Linear(candidate_raw.shape[1],self.dim))

    def candidate(self,raw,ids):
        v=self.candidate_text(raw.float())+self.category(self.cats[ids])+self.subcategory(self.subs[ids])
        v=self.news_norm(F.gelu(v))
        return v,self.bias(v).squeeze(-1)

    def forward(self,hid,cid):
        h,_=self.news(hid)
        u=self.user(h,hid)
        v,b=self.candidate(self.candidate_raw[cid],cid)
        return self.combine(v@u.transpose(1,2),b)
