"""Fetch an immediate slice summary while the larger dev logits are still syncing."""
import argparse
from colab_jobs import ART,NAMES,atomic_json,rex,response

def main():
    p=argparse.ArgumentParser();p.add_argument("session",choices=NAMES);p.add_argument("tag");a=p.parse_args()
    if not a.tag.replace("_","").isalnum():raise ValueError(a.tag)
    code=f"""
import sys,json,numpy as np
from pathlib import Path
r=Path('/content/mind');sys.path.insert(0,str(r));from ranking import arr,metric_rows,NAMES
b=r/'compact';o=arr(b,'dev','offset');y=arr(b,'dev','label');c=arr(b,'dev','candidate')
s=np.load(r/'cache/{a.tag}_dev.npy',mmap_mode='r')
seen=np.zeros(len(np.load(b/'cat.npy')),bool);seen[arr(b,'train','candidate')]=True;m=~seen[c]
n=np.add.reduceat(m,o[:-1],dtype=np.int32);pos=np.add.reduceat(m*y,o[:-1],dtype=np.int32)
good=np.flatnonzero((pos>0)&(pos<n))
v,_=metric_rows(np.r_[0,np.cumsum(n,dtype=np.int64)],y[m],s[m],c[m],seen,good)
q=json.loads((r/'export/{a.tag}_metrics.json').read_text())
q['cold_candidates_only']=dict(impressions=len(good),**dict(zip(NAMES,v.mean(0).tolist())))
print('@@MIND '+json.dumps(q))
"""
    result=response(rex(a.session,code,65,2));atomic_json(ART/"local"/(a.tag+"_dev_slices.json"),result)
    print(result["all"],result["cold_candidates_only"]["AUC"],result["history_le_5"]["AUC"])
if __name__=="__main__":main()

