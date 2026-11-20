"""Common ranking utilities for impression-preserving training and reporting."""
import json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from ranking import metric_rows,NAMES

def record(path,obj):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)

def zscore(values,counts):
    off=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    x=np.asarray(values,dtype=np.float32)
    mean=np.add.reduceat(x,off[:-1],dtype=np.float64)/counts
    centered=x-np.repeat(mean.astype(np.float32),counts)
    sd=np.sqrt(np.add.reduceat(centered*centered,off[:-1],dtype=np.float64)/counts)
    return centered/np.repeat(np.maximum(sd,1e-6).astype(np.float32),counts)

def fit_tree(x,labels,counts,names,rounds,init=None):
    data=lgb.Dataset(x,label=labels,group=counts,feature_name=list(names),init_score=init,free_raw_data=True)
    return lgb.train(dict(objective="lambdarank",metric="None",learning_rate=.065,num_leaves=23,
                min_data_in_leaf=220,max_bin=63,num_threads=4,seed=1119,verbosity=-1,
                deterministic=True,force_col_wise=True,lambdarank_truncation_level=300),data,num_boost_round=rounds)

def row_metrics(labels,counts,scores):
    off=np.r_[0,np.cumsum(counts,dtype=np.int64)]
    dummy=np.zeros(len(labels),np.uint8)
    v,_=metric_rows(off,labels,np.asarray(scores,np.float32),dummy,np.ones(1,bool),np.arange(len(counts)))
    return v

def summarized(total,n):
    return dict(impressions=int(n),**dict(zip(NAMES,(total/n).tolist())))

