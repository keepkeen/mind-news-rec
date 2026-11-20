"""Chronological exposure lookups. Production features never inspect events at or after t."""
import argparse
from functools import lru_cache
import json
from pathlib import Path
import numpy as np
from numba import njit
from ranking import arr

BASE=Path("artifacts/compact")
INDEX=Path("artifacts/exposure")
# 14 days fit in 21 bits; raw user ID and dense news ID form the upper bits.
ORIGIN=1573257600
TIME_BITS=21
TIME_UNIT=1<<TIME_BITS
ITEM_BITS=18

@lru_cache(maxsize=2)
def unpacked_index(path,stamp):
    import zstandard
    with path.open("rb") as f,zstandard.ZstdDecompressor().stream_reader(f) as stream:
        data=np.lib.format.read_array(stream,allow_pickle=False)
    data.flags.writeable=False
    return data

def index_array(name):
    raw=INDEX/(name+".npy")
    try:return np.load(raw,mmap_mode="r")
    except FileNotFoundError:
        packed=raw.with_suffix(".npy.zst")
        return unpacked_index(packed,packed.stat().st_mtime_ns)

def build():
    INDEX.mkdir(parents=True,exist_ok=True)
    splits=["train","dev","test"]
    offsets=[arr(BASE,s,"offset") for s in splits]
    n=sum(int(o[-1]) for o in offsets)
    info={}
    for mode in ["article","user_article"]:
        dst=INDEX/(mode+".npy")
        tmp=dst.with_suffix(".partial")
        data=np.lib.format.open_memmap(tmp,mode="w+",dtype="int64",shape=(n,))
        pos=0
        for s,o in zip(splits,offsets):
            cand=arr(BASE,s,"candidate");users=arr(BASE,s,"user");uids=arr(BASE,s,"user_id")
            times=arr(BASE,s,"time")
            if times.min()<ORIGIN or times.max()-ORIGIN>=TIME_UNIT or cand.max()>=1<<ITEM_BITS: raise ValueError(s)
            for st in range(0,len(o)-1,25000):
                en=min(st+25000,len(o)-1);lo,hi=o[st],o[en]
                reps=np.diff(o[st:en+1]);t=np.repeat(times[st:en]-ORIGIN,reps)
                a=cand[lo:hi].astype(np.int64)
                if mode=="user_article":
                    a+=np.repeat(uids[users[st:en]].astype(np.int64)<<ITEM_BITS,reps)
                data[pos:pos+len(a)]=(a<<TIME_BITS)+t
                pos+=len(a)
        data.sort(kind="quicksort");data.flush();del data;tmp.replace(dst)
        info[mode]=n
        print("indexed",mode,n,flush=True)
    articles=index_array("article")
    keys=np.arange(len(np.load(BASE/"cat.npy"))+1,dtype=np.int64)<<TIME_BITS
    np.save(INDEX/"article_offset.npy",np.searchsorted(articles,keys))
    build_user_offsets()
    (INDEX/"ready.json").write_text(json.dumps(info))

def build_user_offsets():
    pairs=index_array("user_article")
    shift=TIME_BITS+ITEM_BITS
    count=(int(pairs[-1])>>shift)+2
    np.save(INDEX/"user_offset.npy",np.searchsorted(pairs,np.arange(count,dtype=np.int64)<<shift))

@njit(cache=True)
def search(a,lo,hi,q):
    while lo<hi:
        mid=(lo+hi)//2
        if a[mid]<q:lo=mid+1
        else:hi=mid
    return lo

PREFIX=("log_prior_exposure","log_exposure_1h","log_exposure_6h","log_exposure_24h","log_age_seconds",
        "log_user_prior_exposure","log_since_user_exposure","repeated","short_long_rate")
MATCH=("log_slate_length","log_history_length","category_share","subcategory_share","entity_match_recent50")

@njit(cache=True)
def prefix_features(rows,off,cand,times,users,uids,hist,hop,cat,sub,ent,ep,article,ap,ua,up):
    count=0
    for i in rows:count+=off[i+1]-off[i]
    out=np.empty((count,14),np.float32);at=0
    # Category and entity shares count clicked history items, handling overlapping entities once per article.
    cat_count=np.zeros(np.max(cat)+1,np.int32);sub_count=np.zeros(np.max(sub)+1,np.int32)
    ent_count=np.zeros(np.max(ent)+1,np.int32)
    for i in rows:
        u=users[i];raw_uid=np.int64(uids[u]);t=np.int64(times[i])-ORIGIN
        hlo,hhi=hop[u],hop[u+1];last=max(hlo,hhi-50);hsize=hhi-last
        for h in hist[last:hhi]:
            cat_count[cat[h]]+=1;sub_count[sub[h]]+=1
            for e in ent[ep[h]:ep[h+1]]:ent_count[e]+=1
        for ix in range(off[i],off[i+1]):
            n=np.int64(cand[ix]);alo,ahi=ap[n],ap[n+1];start=n<<TIME_BITS
            until=search(article,alo,ahi,start+t)
            age=t-(article[alo]-start) if until>alo else 0
            p1=until-search(article,alo,until,start+t-3600)
            p6=until-search(article,alo,until,start+t-21600)
            p24=until-search(article,alo,until,start+t-86400)
            pair=((raw_uid<<ITEM_BITS)+n)<<TIME_BITS
            # Limit binary search to this user's range; arithmetic uses signed int64.
            left=search(ua,up[raw_uid],up[raw_uid+1],pair)
            right=search(ua,left,up[raw_uid+1],pair+t)
            prev=t-(ua[right-1]-pair) if right>left else 0
            strength=0
            for e in ent[ep[n]:ep[n+1]]:
                strength=max(strength,ent_count[e])
            out[at,0]=np.log1p(until-alo);out[at,1]=np.log1p(p1)
            out[at,2]=np.log1p(p6);out[at,3]=np.log1p(p24);out[at,4]=np.log1p(age)
            out[at,5]=np.log1p(right-left);out[at,6]=np.log1p(prev);out[at,7]=right>left
            out[at,8]=np.log((p1+2)/((p6-p1)/5.+2))
            out[at,9]=np.log1p(off[i+1]-off[i]);out[at,10]=np.log1p(hhi-hlo)
            out[at,11]=cat_count[cat[n]]/max(1,hsize)
            out[at,12]=sub_count[sub[n]]/max(1,hsize);out[at,13]=strength/max(1,hsize)
            at+=1
        for h in hist[last:hhi]:
            cat_count[cat[h]]=0;sub_count[sub[h]]=0
            for e in ent[ep[h]:ep[h+1]]:ent_count[e]=0
    return out

class Reader:
    def __init__(self,split):
        self.split=split
        self.off,self.cand,self.t,self.u,self.uid,self.hist,self.ho=[arr(BASE,split,n) for n in ("offset","candidate","time","user","user_id","hist","hist_offset")]
        self.labels=arr(BASE,split,"label") if split!="test" else None
        self.meta=[np.load(BASE/(n+".npy"),mmap_mode="r") for n in ("cat","subcat","entity","entity_ptr")]
        self.article,self.ap,self.ua,self.up=[index_array(n) for n in ("article","article_offset","user_article","user_offset")]
    def prefix(self,rows):
        return prefix_features(rows,self.off,self.cand,self.t,self.u,self.uid,self.hist,self.ho,*self.meta,self.article,self.ap,self.ua,self.up)
    def positions(self,rows):
        lengths=self.off[rows+1]-self.off[rows]
        cum=np.cumsum(lengths,dtype=np.int64)-lengths
        return np.arange(int(lengths.sum()))+np.repeat(self.off[rows]-cum,lengths)

if __name__=="__main__": build()

