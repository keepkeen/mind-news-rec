"""MIND rank writer, full raw-TSV validator and impression-mean metrics."""
from pathlib import Path
from itertools import zip_longest
import argparse
import io
import json
import os
import zipfile

import numpy as np
from numba import njit

NAMES = ("AUC", "MRR", "nDCG@5", "nDCG@10")

def arr(base, split, name):
    return np.load(Path(base)/split/(name+".npy"), mmap_mode="r")

def rank(scores):
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.int32)
    ranks[order] = np.arange(1, len(scores)+1)
    return ranks

def checked_scores(base, split, scores):
    off = arr(base, split, "offset")
    if scores.shape != (int(off[-1]),) or not np.isfinite(scores).all():
        raise ValueError(f"Wrong score length or nonfinite scores for {split}: {scores.shape}")
    return off

def pack(base, split, scores, path):
    off = checked_scores(base, split, scores)
    ids = arr(base, split, "imp_id")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".incomplete")
    with zipfile.ZipFile(temp,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=2) as z:
        with z.open("prediction.txt","w",force_zip64=True) as out:
            for i, imp in enumerate(ids):
                ranks = rank(scores[off[i]:off[i+1]])
                out.write((str(imp)+" ["+",".join(map(str, ranks.tolist()))+"]\n").encode("ascii"))
    os.replace(temp,path)

def validate(path, raw):
    total = nc = 0
    with zipfile.ZipFile(path) as archive, open(raw,"rb") as src:
        if archive.namelist() != ["prediction.txt"]:
            raise ValueError(f"Unexpected ZIP entries: {archive.namelist()}")
        with archive.open("prediction.txt") as pred:
            for line, row in zip_longest(pred,src):
                if line is None or row is None:
                    raise ValueError(f"Number of lines differs at {total+1}")
                expected, _, _, _, slate = row.rstrip(b"\r\n").split(b"\t")
                fields=line.split()
                if len(fields)!=2:raise ValueError(f"Expected exactly two fields at line {total+1}")
                imp, values = fields
                ranks = json.loads(values)
                n = len(slate.split())
                if imp != expected or type(ranks) is not list or len(ranks)!=n or any(type(r) is not int for r in ranks) or sorted(ranks)!=list(range(1,n+1)):
                    raise ValueError(f"Invalid ID or ranks at line {total+1}")
                total += 1
                nc += n
    return dict(rows=total, candidates=nc, verified_against=str(raw), zip=str(path))

@njit(cache=True)
def metric_rows(offset, label, scores, candidates, seen, rows):
    result = np.empty((len(rows),4), np.float64)
    cold = np.empty(len(rows), np.bool_)
    for j in range(len(rows)):
        i = rows[j]
        lo, hi = offset[i], offset[i+1]
        order = np.argsort(-scores[lo:hi], kind="mergesort")
        p = 0
        wins = mrr = d5 = d10 = ideal5 = ideal10 = 0.0
        cold_count = 0
        for k, idx in enumerate(order):
            cold_count += not seen[candidates[lo+idx]]
            if label[lo+idx]:
                p += 1
                mrr += 1.0/(k+1)
                if k < 10:
                    gain=1.0/np.log2(k+2)
                    d10+=gain
                    if k<5: d5+=gain
            else: wins+=p
        if p==0 or p==hi-lo:
            raise ValueError("An evaluated impression has only one label class")
        for k in range(min(p,10)):
            gain=1.0/np.log2(k+2)
            ideal10+=gain
            if k<5: ideal5+=gain
        result[j,0]=wins/(p*(hi-lo-p))
        result[j,1]=mrr/p
        result[j,2]=d5/ideal5
        result[j,3]=d10/ideal10
        cold[j]=2*cold_count >= hi-lo
    return result,cold

def user_half(user_ids):
    # Odd multiplier mixes low and high bits, splitting by user rather than impression.
    u = np.asarray(user_ids, dtype=np.uint64)
    h = (u ^ (u >> 16))*np.uint64(0x45d9f3b)
    h = (h ^ (h >> 16))*np.uint64(0x45d9f3b)
    return ((h ^ (h >> 16)) & 1).astype(bool)

def metrics(base, split, scores, rows=None, seen=None):
    off = checked_scores(base, split, scores)
    rows = np.arange(len(off)-1) if rows is None else np.asarray(rows)
    cand = arr(base, split, "candidate")
    if seen is None:
        seen=np.zeros(len(np.load(Path(base)/"cat.npy")), bool)
        seen[arr(base, "train","candidate")]=True
    v,cold = metric_rows(off,arr(base,split,"label"),scores,cand,seen,rows)
    u=arr(base,split,"user")[rows]
    lengths=np.diff(arr(base,split,"hist_offset"))[u]
    h=user_half(arr(base,split,"user_id")[u])
    subsets={"all":np.ones(len(rows),bool),"cold_majority":cold,"history_le_5":lengths<=5,
             "user_half_fit":~h,"user_half_check":h}
    out={name:dict(impressions=int(mask.sum()), **dict(zip(NAMES,v[mask].mean(0).tolist())))
         for name,mask in subsets.items() if mask.any()}
    # Restrict the slate itself to unseen candidate IDs; only mixed-label slates have a defined AUC.
    labels=arr(base,split,"label"); novel=~seen[cand]
    n=np.add.reduceat(novel,off[:-1],dtype=np.int32)
    pos=np.add.reduceat(novel*labels,off[:-1],dtype=np.int32)
    good=rows[(pos[rows]>0)&(pos[rows]<n[rows])]
    if len(good):
        co=np.r_[0,np.cumsum(n,dtype=np.int64)]
        vc,_=metric_rows(co,labels[novel],scores[novel],cand[novel],seen,good)
        out["cold_candidates_only"]=dict(impressions=len(good),**dict(zip(NAMES,vc.mean(0).tolist())))
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("action",choices=["trivial","official","metrics","pack","validate"])
    p.add_argument("--base",default="artifacts/compact")
    p.add_argument("--split",choices=["train","dev","test"],default="dev")
    p.add_argument("--scores")
    p.add_argument("--zip",default="artifacts/local/trivial_dev.zip")
    p.add_argument("--out")
    p.add_argument("--expected",help="JSON containing independent metrics for the official roundtrip")
    p.add_argument("--expected-key",default="",help="Optional top-level model key in the metrics JSON")
    a=p.parse_args()
    if a.split=="test" and a.action not in ("pack","validate"):
        p.error("Test has no labels. Supported test operations are pack and validate.")
    path=Path(a.zip)
    raw=Path("data")/("large_"+a.split)/("MINDlarge_"+a.split)/"behaviors.tsv"
    if a.action=="validate":
        result=validate(path,raw)
    elif a.action=="official":
        import sys
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
        import evaluate
        def truth():
            with raw.open("rb") as f:
                for line in f:
                    imp,_,_,_,c=line.rstrip().split(b"\t")
                    labels=[int(x[-1:]) for x in c.split()]
                    yield f"{imp.decode()} {json.dumps(labels,separators=(',',':'))}\n"
        with zipfile.ZipFile(path) as z, z.open("prediction.txt") as f:
            result=dict(zip(NAMES,evaluate.scoring(truth(),io.TextIOWrapper(f,encoding="ascii"))))
        if a.expected:
            ours=json.loads(Path(a.expected).read_text())
            if a.expected_key:ours=ours[a.expected_key]
            ours=ours.get("all",ours)
        else:
            ours=json.loads(Path("artifacts/local/trivial_dev.json").read_text())["metrics"]["all"]
        result["max_abs_diff"]=max(abs(result[m]-ours[m]) for m in NAMES)
        if result["max_abs_diff"]>1e-10: raise ValueError(result)
    elif a.action=="trivial":
        size=int(arr(a.base,a.split,"offset")[-1])
        scores=np.random.default_rng(20261119).random(size,dtype=np.float32)
        pack(a.base,a.split,scores,path)
        result=dict(metrics=metrics(a.base,a.split,scores),validation=validate(path,raw),seed=20261119)
    else:
        scores=np.load(a.scores,mmap_mode="r")
        if a.action=="pack":
            pack(a.base,a.split,scores,path)
            result=validate(path,raw)
        else: result=metrics(a.base,a.split,scores)
    s=json.dumps(result,indent=2)
    if a.out:
        dst=Path(a.out);dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(s)
    print(s,flush=True)

if __name__=="__main__": main()

