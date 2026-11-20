"""Exact rolling checkpoints for an interrupted epoch of click-supervised training."""
from pathlib import Path
import torch
from jobio import CACHE,export_file

def snapshot(job,ep,step,permutation_rng,encoder,head,opt,schedule,spec,seed,best,best_ep):
    name=f"{job}_rolling_e{ep}_s{step:04d}.pt"
    out=CACHE/name;partial=out.with_suffix(".saving")
    torch.save(dict(job=job,epoch=ep,step=step,hf_revision=spec["revision"],seed=seed,spec=spec,
        article_trainable=encoder.save_trainable(),head=head.state_dict(),optimizer=opt.state_dict(),scheduler=schedule.state_dict(),
        permutation_rng=permutation_rng,cuda_rng=torch.cuda.get_rng_state(),cpu_rng=torch.get_rng_state(),best=best,best_epoch=best_ep),partial)
    partial.replace(out);export_file(out)
    # Remote export chunks are retained until a later locally verified snapshot supersedes them.
    out.unlink()

def restore(path,job,encoder,head,opt,schedule,spec,seed):
    saved=torch.load(Path(path),map_location="cpu",weights_only=True)
    if saved["job"]!=job or saved["seed"]!=seed or saved["hf_revision"]!=spec["revision"]:raise ValueError("Rolling checkpoint belongs to a different run")
    for k in ("training_splits","positive_pairs","logical_positives","max_tokens","epochs","shadow"):
        if spec[k]!=saved["spec"][k]:raise ValueError("Resume protocol mismatch: "+k)
    with torch.no_grad():
        params={n:p for n,p in encoder.named_parameters() if p.requires_grad}
        if params.keys()!=saved["article_trainable"].keys():raise ValueError("Incomplete encoder state")
        for n,p in params.items():p.copy_(saved["article_trainable"][n])
        head.load_state_dict(saved["head"],strict=True)
    opt.load_state_dict(saved["optimizer"]);schedule.load_state_dict(saved["scheduler"])
    torch.set_rng_state(saved["cpu_rng"]);torch.cuda.set_rng_state(saved["cuda_rng"])
    return saved["epoch"],saved["step"],saved["permutation_rng"],saved["best"],saved["best_epoch"]
