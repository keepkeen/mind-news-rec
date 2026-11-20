"""Exercise threshold boundaries and the complete release path on a real 2,048-impression slice."""
from contextlib import ExitStack,redirect_stdout
from io import StringIO
from itertools import islice
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import numpy as np
import fusion
import incremental as pub
import incremental_policy as policy
from colab_jobs import ART,ROOT,atomic_json
from ranking import arr,validate

def main():
    real=policy.catalog()
    base=real[frozenset()]
    core=frozenset(["fullbge"])
    t={**base,"full_dev":policy.MIN_DEV,"selection_A":.728,"heldout_B":base["heldout_B"]}
    examples={frozenset():base,core:t}
    gate=policy.assess(core,examples)
    assert gate["release"]
    for field,n in [("full_dev",np.nextafter(policy.MIN_DEV,-np.inf)),("heldout_B",np.nextafter(base["heldout_B"],-np.inf))]:
        assert not policy.assess(core,{**examples,core:{**t,field:n}})["release"]
    extra=core|{"upper_s2"}
    assert not policy.assess(extra,{**examples,extra:{**t,"selection_A":t["selection_A"]}})["release"]
    assert policy.assess(extra,{**examples,extra:{**t,"selection_A":t["selection_A"]+.0001}})["release"]
    with TemporaryDirectory(dir=policy.R5,prefix="smoke_") as temp:
        root=Path(temp);ids=root/"compact/test";ids.mkdir(parents=True)
        src=ART/"compact";r=fusion.Reader("test");num=2048
        r.off=r.off[:num+1];r.cand=r.cand[:int(r.off[-1])]
        np.save(ids/"offset.npy",r.off);np.save(ids/"imp_id.npy",arr(src,"test","imp_id")[:num])
        raw=root/"first_test_rows.tsv"
        with (ROOT/"data/large_test/MINDlarge_test/behaviors.tsv").open("rb") as f,raw.open("wb") as w:
            w.writelines(islice(f,num))
        # The established three-model policy exercises real early shadows and matched final checkpoints.
        atomic_json(root/"policy.json",policy.read(ART/"fusion/candidate_policy.json"))
        fusion.refit_tree(root)
        with patch.object(fusion,"Reader",return_value=r),patch.object(fusion,"BASE",ids.parent),patch.object(fusion,"validate",lambda p,_:validate(p,raw)):
            fusion.score_test(root,"policy.json","ranker.lgb","release_smoke",refit_base=True,output_dir=root)
        z=root/"release_smoke.zip";rp=z.with_suffix(".validation.json");receipt=policy.read(rp)
        assert receipt["rows"]==num and receipt["candidates"]==len(r.cand) and len(receipt["matched_refit"])==3
        # Simulate a qualifying dev gate only inside the temporary publication directory.
        ledger=pub.empty_manifest()
        spec=dict(key="a"*64,gate=gate,trial=t,refs=receipt["matched_refit"])
        receipt.update(incremental_key=spec["key"],metrics={k:t[k] for k in ("full_dev","selection_A","heldout_B")})
        atomic_json(rp,receipt)
        with ExitStack() as stack:
            stack.enter_context(redirect_stdout(StringIO()))
            for obj,name,value in [(pub,"SUB",root/"sub"),(pub,"MANIFEST",root/"book.json"),(pub,"R5",root),
                                   (pub,"notify",lambda *_:True),(policy,"TEST_SHAPE",(num,len(r.cand)))]:
                stack.enter_context(patch.object(obj,name,value))
            pub.register(spec,z,ledger);pub.register(spec,z,ledger)
            assert len(ledger["packages"])==1 and policy.sha256(root/"sub/fusion_v2_prefix.zip")==receipt["sha256"]
            inferior={**t,"selection_A":t["selection_A"]-.0001}
            second={**spec,"key":"b"*64,"trial":inferior}
            receipt.update(incremental_key=second["key"],metrics={k:inferior[k] for k in ("full_dev","selection_A","heldout_B")})
            atomic_json(rp,receipt);pub.register(second,z,ledger)
            assert len(ledger["packages"])==2 and ledger["preferred"]["v2"]==spec["key"]
            receipt["sha256"]="0"*64;atomic_json(rp,receipt)
            try:policy.validated_copy(z,root/"invalid.zip","integrity smoke")
            except ValueError:pass
            else:raise AssertionError("Wrong digest accepted")
        out=dict(real_test_rows=num,real_candidates=len(r.cand),refit_tree_days="Nov 12–15",local_rank_zip_sha256=policy.sha256(z),
                 real_matched_epochs={k:v["epoch"] for k,v in spec["refs"].items()},inclusive_threshold=True,B_failure_rejected=True,
                 nonpositive_seed_A_rejected=True,immutable_archive_and_alias=True,restart_idempotent=True,
                 lower_A_did_not_replace_alias=True,invalid_sha_rejected=True,production_files_published_by_smoke=0)
        atomic_json(policy.R5/"pipeline_smoke.json",out)
        print(out)
if __name__=="__main__":main()
