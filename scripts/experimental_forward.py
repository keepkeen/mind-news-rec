"""Build the isolated offline six-branch future-exposure counterfactual on CPU."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
import zipfile
import lightgbm as lgb
import numpy as np
from experimental_exposure import DAY, FORWARD, Timeline, values
from exposures import BASE, Reader
from fusion import feature, grid_auc, sample, weights
from independent_submission import TEST_SHAPE,verify
from ranking import arr, metrics, rank, user_half
from trees import fit_tree, record, row_metrics, zscore

WORK = Path("artifacts/experimental_forward")
ARCHIVE = Path("artifacts/submissions/experimental_forward.zip")
FORMAL = Path("artifacts/submissions/final_prefix.zip")
AUDIT = Path("artifacts/round7/final_audit.json")
RAW = Path("data/large_test/MINDlarge_test/behaviors.tsv")

@dataclass(frozen=True)
class ForwardRun:
    work: Path = WORK
    archive: Path = ARCHIVE
    formal: Path = FORMAL
    raw: Path = RAW
    test_shape: tuple = TEST_SHAPE
    daily_groups: int = 40000
    dev_limit: int | None = None
    locked_weights: bool = False

    def progress(self, stage, **kwargs):
        record(self.work/"progress.json",dict(stage=stage,utc=time.time(),**kwargs))

def read(path):
    return json.loads(Path(path).read_text())

def digest(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()

def load_inputs(audit, stage, reader):
    scores = []
    for name in audit["models"]:
        branch = audit["branches"][name]
        path = Path(branch["files"][stage])
        if Path(str(path) + ".verified").read_text().strip() != branch["hashes"][stage]:
            raise ValueError(f"Input digest does not match the formal audit: {path}")
        scores.append(np.load(path, mmap_mode="r"))
        if scores[-1].shape != (int(reader.off[-1]),):
            raise ValueError(f"Misaligned logits: {path}")
    return scores

def training_rows(r, groups_per_day=40000):
    groups = []
    for day in (12, 13, 14):
        start = int(np.datetime64(f"2019-11-{day}T00:00:00").astype("int64"))
        groups.append(sample(np.flatnonzero((r.t >= start) & (r.t < start + DAY)), groups_per_day, 1100 + day))
    return np.sort(np.concatenate(groups))

class Distribution:
    def __init__(self):
        self.parts = []

    def collect(self, matrix):
        self.parts.append(matrix[::251].copy())

    def report(self):
        x = np.concatenate(self.parts)
        q = np.quantile(x, (0.5, 0.9), axis=0)
        avg = x.mean(axis=0, dtype=np.float64)
        zero = (x == 0).mean(axis=0)
        return dict(sampled_candidates=len(x), features={
            n: dict(mean=float(avg[i]), median=float(q[0,i]), p90=float(q[1,i]), zero_share=float(zero[i]))
            for i,n in enumerate(FORWARD)})

def matrix(reader, rows, scores, horizon, timeline, distribution):
    base, z, sizes, pos = feature(reader, rows, scores)
    forward = values(reader, rows, horizon, timeline)
    distribution.collect(forward)
    return np.column_stack((base, forward)), z, sizes, pos

def training_matrix(reader, rows, scores, horizon, timeline, cols, dist):
    counts = reader.off[rows + 1] - reader.off[rows]
    length = int(counts.sum())
    x = np.empty((length, len(cols)), np.float32)
    y = np.empty(length, reader.labels.dtype)
    at = 0
    for st in range(0, len(rows), 6000):
        group = rows[st:st+6000]
        xx, _, _, pos = matrix(reader, group, scores, horizon, timeline, dist)
        end = at + len(pos)
        x[at:end], y[at:end] = xx, reader.labels[pos]
        at = end
    return x, y, counts

def blend(z, tree, w):
    result = tree * w[-1]
    for i in np.flatnonzero(w[:-1]):
        result += z[:,i] * w[i]
    return result

def scores_for_weights(reader, tree_z, inputs, w, n_dev):
    result = tree_z * w[-1]
    for st in range(0, n_dev, 8000):
        end = min(st+8000, n_dev)
        sl = slice(reader.off[st], reader.off[end])
        sizes = np.diff(reader.off[st:end+1])
        for i in np.flatnonzero(w[:-1]):
            result[sl] += zscore(inputs[i][sl], sizes) * w[i]
    return result

def search_a(reader, scores, tree_z, official_weights):
    half = user_half(reader.uid[reader.u])
    rows = sample(np.flatnonzero(~half), 30000, 1120)
    pos = reader.positions(rows)
    sizes = reader.off[rows+1] - reader.off[rows]
    z = np.column_stack([*[zscore(v[pos], sizes) for v in scores], tree_z[pos]])
    off = np.r_[0, np.cumsum(sizes, dtype=np.int64)]
    y = reader.labels[pos]
    best = -1.0
    for w in weights(z.shape[1]):
        auc = grid_auc(y, off, np.sum(z * w, axis=1))
        if auc > best:
            selected, best = w, float(auc)
    fixed = float(grid_auc(y, off, np.sum(z * official_weights, axis=1)))
    return selected, dict(tune_groups=len(rows), tune_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
                          search_auc=best, fixed_auc=fixed, grid_weights=selected.tolist(),
                          package_weights=official_weights.tolist(), sampled_users="A only",
                          package_policy="predeclared official six and official blend weights")

def compare(r, baseline, proposed, horizon):
    n=len(horizon)
    sizes = np.diff(r.off[:n+1])
    labels=r.labels[:r.off[n]]
    deltas = row_metrics(labels, sizes, proposed)[:,0] - row_metrics(labels, sizes, baseline)[:,0]
    halves = user_half(r.uid[r.u[:n]])
    result = {}
    for key,mask in (("A",~halves),("B",halves),("observed_under_24h",horizon<DAY),
                     ("observed_at_least_24h",horizon>=DAY)):
        if mask.any():result[key] = dict(impressions=int(mask.sum()), paired_auc_delta=float(deltas[mask].mean()))
    return result

def dev_report(reader,scores,n_dev):
    if n_dev==len(reader.t):return metrics(BASE,"dev",scores)
    # The rehearsal evaluates only these genuine groups with the same impression metric routine.
    padded=np.zeros(len(reader.cand),np.float32)
    padded[:len(scores)]=scores
    return metrics(BASE,"dev",padded,rows=np.arange(n_dev))

def fit_and_evaluate(audit, source, tl, columns, fixed, run):
    train = Reader("train")
    hr = tl.horizons("train")
    rows = training_rows(train,run.daily_groups)
    train_dist, dev_dist = Distribution(), Distribution()
    run.progress("fit_tree_on_Nov_12_14", groups=len(rows))
    xt, yt, gt = training_matrix(train, rows, load_inputs(audit,"shadow",train), hr, tl, columns, train_dist)
    model = fit_tree(xt, yt, gt, columns, 100)
    model.save_model(str(run.work / "dev_tree.lgb"))
    official = lgb.Booster(model_file=str(source / "temporal_tree.lgb"))
    formal_metrics = read(source / "dev_metrics.json")["blend"]
    dev = Reader("dev")
    n_dev=len(dev.t) if run.dev_limit is None else run.dev_limit
    if not 1<=n_dev<=len(dev.t):raise ValueError("Invalid dev count")
    hd = tl.horizons("dev")
    ds = load_inputs(audit,"dev",dev)
    tree_z = np.empty(int(dev.off[n_dev]), np.float32)
    reference = np.empty_like(tree_z)
    for st in range(0, n_dev, 6000):
        end = min(st+6000, n_dev)
        pick = np.arange(st, end)
        x, z, sizes, _ = matrix(dev,pick,ds,hd,tl,dev_dist)
        sl = slice(dev.off[st], dev.off[end])
        tree_z[sl] = zscore(model.predict(x, num_threads=4), sizes)
        tree_base = zscore(official.predict(x[:,:official.num_feature()], num_threads=4), sizes)
        reference[sl] = blend(z, tree_base, fixed)
        run.progress("score_official_dev", done=end, total=n_dev)
    if run.locked_weights:
        selected=fixed
        selection=dict(package_policy="exact branch order and weights of the published official archive",
                       package_weights=fixed.tolist(),weight_search_performed=False)
    else:
        run.progress("select_on_A")
        selected, selection = search_a(dev, ds, tree_z, fixed)
    record(run.work / "weights_from_A.json", selection)
    proposed = scores_for_weights(dev, tree_z, ds, fixed, n_dev)
    base = dev_report(dev,reference,n_dev)
    if any(abs(base[k][m]-v[m])>1e-11 for k,v in formal_metrics.items() for m in ("AUC","MRR","nDCG@5","nDCG@10")):
        raise ValueError("Paired control did not reproduce the locked official dev metrics")
    experiment = dev_report(dev,proposed,n_dev)
    grid_report = experiment if np.array_equal(selected,fixed) else dev_report(dev,scores_for_weights(dev,tree_z,ds,selected,n_dev),n_dev)
    gate = audit["gate"] if "gate" in audit else read("artifacts/round7/offline_lock.json")["selected"]["gate"]
    b = experiment["user_half_check"]["AUC"]
    checks = dict(B_beats_locked_control=b >= base["user_half_check"]["AUC"],
                  B_beats_formal_reference=b >= gate["B_reference"], B_reference=gate["B_reference"])
    if not run.locked_weights:
        checks["A_grid_B_beats_locked_control"]=grid_report["user_half_check"]["AUC"] >= base["user_half_check"]["AUC"]
    report = dict(formal=base, experimental_fixed=experiment, checks=checks,
                  delta={k:experiment[k]["AUC"]-base[k]["AUC"] for k in base},
                  paired=compare(dev,reference,proposed,hd[:n_dev]), selection=selection)
    if not run.locked_weights:report["A_grid"]=grid_report
    record(run.work / "dev_metrics.json",report)
    print("DEV",json.dumps(dict(full=experiment["all"], delta=report["delta"], weights=selection)),flush=True)
    coverage = dict(train_fit=tl.coverage(train.t[rows],hr[rows]), dev=tl.coverage(dev.t[:n_dev],hd[:n_dev]),
                    test_hour_by_remaining_days=tl.by_hour.tolist(), end_inclusive=tl.end)
    record(run.work / "coverage.json",coverage)
    distributions = dict(train_fit=train_dist.report(),dev=dev_dist.report())
    record(run.work / "feature_distributions.json",distributions)
    run.progress("refit_with_out_of_sample_Nov_15")
    dr = sample(np.arange(n_dev),run.daily_groups,1115)
    xd, yd, gd = training_matrix(dev,dr,ds,hd,tl,columns,Distribution())
    final = fit_tree(np.concatenate((xt,xd)),np.concatenate((yt,yd)),np.concatenate((gt,gd)),columns,100)
    final.save_model(str(run.work / "test_tree.lgb"))
    record(run.work/"training.json",dict(fit_days=[12,13,14],final_tree_days=[12,13,14,15],
          groups_per_day=run.daily_groups, fit_groups=len(rows), final_fit_groups=len(rows)+len(dr),
          train_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
          dev_refit_rows_sha256=hashlib.sha256(dr.tobytes()).hexdigest(),
          dev_tree_sha256=digest(run.work/"dev_tree.lgb"), final_tree_sha256=digest(run.work/"test_tree.lgb")))

def write_test(audit, tl, fixed, run):
    test = Reader("test")
    inputs = load_inputs(audit,"test",test)
    ht = tl.horizons("test")
    n_test,items=run.test_shape
    if int(test.off[n_test])!=items:raise ValueError("Test candidate count does not match the evaluation scope")
    coverage = read(run.work / "coverage.json")
    coverage["test"] = tl.coverage(test.t[:n_test],ht[:n_test])
    record(run.work / "coverage.json",coverage)
    model = lgb.Booster(model_file=str(run.work/"test_tree.lgb"))
    dist = Distribution()
    ids = arr(BASE,"test","imp_id")
    staging = run.work/(run.archive.stem+".staging.zip")
    begin = time.monotonic()
    with zipfile.ZipFile(staging,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=2) as zf:
        with zf.open("prediction.txt","w",force_zip64=True) as out:
            for st in range(0,n_test,6000):
                end = min(st+6000,n_test)
                x, z, sizes, _ = matrix(test,np.arange(st,end),inputs,ht,tl,dist)
                pred = blend(z,zscore(model.predict(x,num_threads=4),sizes),fixed)
                if not np.isfinite(pred).all():
                    raise ValueError(f"Nonfinite test score at group {st}")
                offsets = test.off[st:end+1] - test.off[st]
                buf = bytearray()
                for j,impid in enumerate(ids[st:end]):
                    ranks = rank(pred[offsets[j]:offsets[j+1]])
                    buf.extend((str(impid)+" ["+",".join(map(str,ranks.tolist()))+"]\n").encode("ascii"))
                out.write(buf)
                run.progress("scoring_and_writing_test",done=end,total=n_test,
                             eta_seconds=(n_test-end)*(time.monotonic()-begin)/end)
    stats = read(run.work/"feature_distributions.json")
    stats["test"] = dist.report()
    record(run.work/"feature_distributions.json",stats)
    run.progress("independent_verification")
    result = verify(staging,run.raw,expected_shape=run.test_shape)
    if digest(run.formal) != audit["final_sha256"]:
        raise ValueError("The pinned official control archive changed")
    run.archive.parent.mkdir(exist_ok=True,parents=True)
    os.replace(staging,run.archive)
    result.update(zip=str(run.archive),future_exposure=True,role="experimental_control",official_sha256=audit["final_sha256"])
    if run.locked_weights:
        result.update(input_signature=audit["input_signature"],models=audit["models"],weights=audit["weights"],
                      content_training_by_branch={n:audit["branches"][n]["training"] for n in audit["models"]})
    record(run.work/"independent_validation.json",result)
    record(run.archive.with_suffix(".validation.json"),result)
    run.progress("DONE",**result)
    print("VERIFIED",json.dumps(result),flush=True)

def run_forward(run, audit_path, source, mode="all", resume=False):
    run.work.mkdir(parents=True,exist_ok=True)
    audit=read(audit_path)
    baseline=read(source/"candidate_policy.json")
    if baseline["type"]!="prefix-only" or audit["models"]!=baseline["models"] or audit["weights"]!=baseline["weights"]:
        raise ValueError("Official policy differs from the pinned audit")
    if digest(run.formal)!=audit["final_sha256"]:
        raise ValueError("Invalid baseline archive")
    fixed=np.asarray(audit["weights"],np.float32)
    tl=Timeline()
    columns=[*baseline["columns"],*FORWARD]
    policy=dict(models=audit["models"],weights=fixed.tolist(),columns=columns,type="experimental-forward",
                official_archive_sha256=audit["final_sha256"],input_sources=str(audit_path),
                timestamps="strictly later, upper inclusive, continuous train to dev to test",
                article_windows_hours=[1,6,24], last_article_and_user_window_hours=168,
                censoring="public test hour x remaining day-count mixture for train/dev; actual recording end for test")
    if run.locked_weights:
        policy.update(input_signature=audit["input_signature"],weight_search_performed=False,
                      daily_groups=run.daily_groups,test_shape=list(run.test_shape),dev_limit=run.dev_limit)
    policy_path=run.work/"policy.json";training=run.work/"training.json";tree=run.work/"test_tree.lgb"
    cached=(resume and policy_path.is_file() and read(policy_path)==policy and training.is_file()
            and (run.work/"dev_metrics.json").is_file() and tree.is_file()
            and digest(tree)==read(training)["final_tree_sha256"])
    if mode in ("all","dev") and not cached:
        record(policy_path,policy)
        fit_and_evaluate(audit,source,tl,columns,fixed,run)
    if mode in ("all","test"):
        if read(policy_path)!=policy or digest(tree)!=read(training)["final_tree_sha256"]:
            raise ValueError("The final experimental tree does not match its fitted policy")
        write_test(audit,tl,fixed,run)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage",choices=("all","dev","test"),default="all")
    source=Path(read("artifacts/round7/offline_lock.json")["selected"]["dest"])
    run_forward(ForwardRun(),AUDIT,source,mode=p.parse_args().stage)

if __name__=="__main__":
    main()
