"""Score the serialized click checkpoint itself for the already-running full BGE and 0.6B fits."""
import hashlib
import json
import sys
import time
import numpy as np
import torch
from click_text import ArticleEncoder,SPECS
from recommender import Model,Split,gpu,score
from jobio import CACHE,DATA,EXPORT,RUN,write_json,export_file,run_job,status

def main(kind):
    role,epochs,seed={"fullbge":("fullfit",3,1125),"qlora":("qfit",2,1126)}[kind]
    job=kind+"_checkpoint_replay"
    status(job,"waiting_for_fit",dependency="fleet_"+role)
    marker=RUN/("fleet_"+role+".status.json")
    while not marker.is_file() or json.loads(marker.read_text())["phase"] not in ("DONE","FAILED"):time.sleep(15)
    torch.manual_seed(seed);torch.set_float32_matmul_precision("high")
    enc=None;test=None;finished=[]
    for ep in range(1,epochs+1):
        tag=f"{kind}_refit_e{ep}"
        rp=EXPORT/(tag+".receipt.json")
        if not rp.is_file():continue
        receipt=json.loads(rp.read_text());metadata=receipt["checkpoint"]
        cp=CACHE/metadata["name"]
        with cp.open("rb") as f:
            if hashlib.file_digest(f,"sha256").hexdigest()!=metadata["sha256"]:raise ValueError(cp)
        checkpoint=torch.load(cp,map_location="cpu",weights_only=True)
        if (checkpoint["epoch"],checkpoint["seed"],tuple(checkpoint["hf"]))!=(ep,seed,SPECS[kind]):raise ValueError(tag)
        spec=checkpoint["spec"]
        if enc is None:
            enc=ArticleEncoder(kind,spec["max_tokens"]);test=Split("test")
            buf=torch.zeros(len(enc.lengths),enc.width,dtype=torch.float16,device="cuda")
            head=Model(buf,gpu(np.load(DATA/"cat.npy").astype(np.int32)),gpu(np.load(DATA/"subcat.npy").astype(np.int32)),4).cuda()
        with torch.no_grad():
            params={n:p for n,p in enc.named_parameters() if p.requires_grad}
            saved=checkpoint["article_trainable"]
            if set(saved)!=set(params):raise ValueError("Missing trainable parameter")
            for n,p in params.items():p.copy_(saved[n])
            head.load_state_dict(checkpoint["head"],strict=True)
            status(job,"scoring_serialized_checkpoint",epoch=ep,epochs=epochs)
            enc.all_articles(buf,spec["encoder_micro"])
            path,_=score(head,test,tag+"_replay");export_file(path)
        m=json.loads((EXPORT/(path.name+".parts.json")).read_text())
        receipt.update(checkpoint_scored=True,test_scores=dict(name=path.name,sha256=m["sha256"],bytes=path.stat().st_size))
        write_json(EXPORT/(tag+".replay.json"),receipt);finished.append(ep)
    status(job,"DONE",epochs=finished)
if __name__=="__main__":
    kind=sys.argv[1];run_job(kind+"_checkpoint_replay",lambda:main(kind))
