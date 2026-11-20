"""Hold out the entire official dev day from tree fitting to quantify same-day transfer."""
from pathlib import Path
import json
import time
import numpy as np
from exposures import Reader,BASE,MATCH,PREFIX
from fusion import feature,sample,branch
from trees import record,fit_tree
from ranking import metrics

def main():
    dest=Path("artifacts/fusion_bge");r=Reader("train");v=Reader("dev")
    shadow=Path("artifacts/jobs/mind_h100/bge_shadow_e2_train.npy")
    assert shadow.with_name(shadow.name+".verified").exists()
    early=np.load(shadow,mmap_mode="r");scores=np.load(branch("bge","dev"),mmap_mode="r")
    start=int(np.datetime64("2019-11-12T00:00:00").astype("int64"))
    rows=sample(np.flatnonzero(r.t>=start),60000,1119)
    x,_,sz,pos=feature(r,rows,[early])
    m=fit_tree(x,r.labels[pos],sz,[*PREFIX,*MATCH,"z_bge"],100)
    m.save_model(str(dest/"temporal_bge.lgb"))
    out=np.empty(len(v.cand),np.float32)
    begin=time.time()
    for st in range(0,len(v.u),8000):
        en=min(st+8000,len(v.u));rows_v=np.arange(st,en)
        x,_,_,_=feature(v,rows_v,[scores]);out[v.off[st]:v.off[en]]=m.predict(x,num_threads=4)
        record(dest/"temporal_progress.json",dict(stage="dev",rows=en,total=len(v.u),eta_sec=(len(v.u)-en)*(time.time()-begin)/en))
    measured=metrics(BASE,"dev",out);base=metrics(BASE,"dev",scores)
    record(dest/"temporal_check.json",dict(tree_fit="60000 complete train impressions on Nov 12–14; no dev labels",
        tree_training_content="BGE-NRMS trained on Nov 9–11",dev_content="BGE-NRMS trained on Nov 9–14",
        metrics=measured,delta_vs_bge_full_dev=measured["all"]["AUC"]-base["all"]["AUC"]))
    record(dest/"temporal_progress.json",dict(stage="DONE"))
if __name__=="__main__":main()

