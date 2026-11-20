"""Check equal-timestamp ties, strict past boundaries, and source-row independence."""
import numpy as np
from exposures import ORIGIN,TIME_BITS,ITEM_BITS,prefix_features
from trees import record
from ranking import rank

def main():
    # Rows are intentionally not in timestamp order; two different slates have t=4000.
    t=np.array([4000,500,4000,8000],np.int32);u=np.array([0,0,1,0],np.int32);uid=np.array([1,2])
    rows=np.array([[1,2],[1,3],[1,3],[1,2]],np.int32);cand=rows.ravel()
    off=np.arange(0,9,2,dtype=np.int32);nt=np.repeat(t,2)
    news=cand.astype(np.int64);raw=np.repeat(uid[u],2)
    articles=np.sort((news<<TIME_BITS)+nt);ap=np.searchsorted(articles,np.arange(5,dtype=np.int64)<<TIME_BITS)
    ua=np.sort((((raw<<ITEM_BITS)+news)<<TIME_BITS)+nt)
    up=np.searchsorted(ua,np.arange(4,dtype=np.int64)<<(ITEM_BITS+TIME_BITS))
    features=prefix_features(np.arange(4),off,cand,t+ORIGIN,u,uid,np.array([1,3]),np.array([0,1,2]),
                              np.array([0,1,1,2]),np.array([0,1,2,3]),np.array([1,2,3]),
                              np.array([0,0,1,2,3]),articles,ap,ua,up)
    for j,n in enumerate(news):
        before=(news==n)&(nt<nt[j]);personal=before&(raw==raw[j])
        expected=[before.sum(),(before&(nt>=nt[j]-3600)).sum(),(before&(nt>=nt[j]-21600)).sum(),
                  (before&(nt>=nt[j]-86400)).sum(),(nt[j]-nt[before].min() if before.any() else 0),
                  personal.sum(),(nt[j]-nt[personal].max() if personal.any() else 0)]
        np.testing.assert_allclose(np.expm1(features[j,:7]),expected,rtol=1e-5,atol=1e-5)
    assert rank(np.array([.5,.5,.7])).tolist()==[2,3,1]
    result={"candidate_events_checked":8,"equal_timestamps":True,"unsorted_input":True,"stable_rank_tie":True}
    record("artifacts/local/prefix_sanity.json",result);print(result)
if __name__=="__main__":main()

