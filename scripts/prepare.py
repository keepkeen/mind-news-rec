"""Integerize the released TSVs, preserving row order and the full user histories."""
import argparse
from array import array
from datetime import date
import hashlib
import json
from pathlib import Path
import time

import numpy as np

SPLITS = ("train", "dev", "test")
BASE = Path("artifacts/compact")

def save(path, name, values, dtype):
    np.save(path / (name + ".npy"), np.asarray(values, dtype=dtype), allow_pickle=False)

def encode_news():
    BASE.mkdir(parents=True, exist_ok=True)
    articles = {}
    for split in SPLITS:
        with (Path("data") / ("large_" + split) / ("MINDlarge_" + split) / "news.tsv").open() as f:
            for line in f:
                nid, cat, subcat, title, abstract, _, title_entities, abstract_entities = line.rstrip("\r\n").split("\t")
                if nid not in articles:
                    entities = [e["WikidataId"] for text in (title_entities, abstract_entities)
                                for e in json.loads(text or "[]") if e.get("WikidataId")]
                    articles[nid] = [nid, title, abstract, cat, subcat, list(dict.fromkeys(entities))]
    keys = sorted(articles, key=lambda x: int(x[1:]))
    cats, subs, ents = {}, {}, {}
    cat_ids, sub_ids, entity_ids, ptr = [0], [0], [], [0, 0]
    with (BASE / "news.jsonl").open("w") as out:
        out.write(json.dumps(["", "", "", "", "", []]) + "\n")
        for nid in keys:
            row = articles[nid]
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            cat_ids.append(cats.setdefault(row[3], len(cats) + 1))
            sub_ids.append(subs.setdefault(row[4], len(subs) + 1))
            entity_ids.extend(ents.setdefault(e, len(ents) + 1) for e in row[5])
            ptr.append(len(entity_ids))
    for name, values, dtype in (("cat", cat_ids, "int16"), ("subcat", sub_ids, "int16"),
                                 ("entity", entity_ids, "int32"), ("entity_ptr", ptr, "int32")):
        save(BASE, name, values, dtype)
    (BASE / "news_meta.json").write_text(json.dumps({"articles":len(keys), "categories":cats, "subcategories":subs,
                             "entity_count":len(ents), "row_zero":"padding", "fields":["news_id","title","abstract","category","subcategory","entity_ids"]}))
    print("news", len(keys), flush=True)

def encode_split(split):
    out = BASE / split
    out.mkdir(parents=True, exist_ok=True)
    with (BASE / "news.jsonl").open() as f:
        mapping = {json.loads(row)[0].encode(): i for i, row in enumerate(f)}
    if split == "test":
        lookup = mapping
    else:
        lookup = {nid + suffix: 2*i+y for nid,i in mapping.items()
                  for y,suffix in enumerate((b"-0", b"-1"))}
    cand = array("I")
    offsets, hist_off = array("I", [0]), array("I", [0])
    imp_ids, times, imp_user = array("I"), array("I"), array("I")
    user_ids, hist = array("I"), array("I")
    users, variants, hist_strings = {}, {}, []
    stamps, dates = {}, {}
    seen = set()
    hist_variant_count = 0
    inp = Path("data") / ("large_" + split) / ("MINDlarge_" + split) / "behaviors.tsv"
    digest = hashlib.sha256()
    with inp.open("rb") as f:
        for count, row in enumerate(f, 1):
            digest.update(row)
            imp, uid, stamp, history, candidates = row.rstrip(b"\r\n").split(b"\t")
            imp_ids.append(int(imp))
            idx = users.get(uid)
            if idx is not None and hist_strings[idx] != history:
                # Preserve a distinct cache row if a release contains multiple histories for one user.
                idx = variants.get((uid, history))
                if idx is None:
                    idx = len(user_ids)
                    variants[(uid, history)] = idx
                    hist_variant_count += 1
            elif idx is None:
                idx = len(user_ids)
                users[uid] = idx
            if idx == len(user_ids):
                user_ids.append(int(uid[1:]))
                hist_strings.append(history)
                hist.extend(map(mapping.__getitem__, history.split()))
                hist_off.append(len(hist))
            imp_user.append(idx)
            ts = stamps.get(stamp)
            if ts is None:
                day, clock, meridian = stamp.split()
                if day not in dates:
                    m, d, y = map(int, day.split(b"/"))
                    dates[day] = (date(y,m,d) - date(1970,1,1)).days * 86400
                h, mi, sec = map(int, clock.split(b":"))
                ts = dates[day] + ((h % 12) + 12*(meridian == b"PM"))*3600 + mi*60 + sec
                stamps[stamp] = ts
            times.append(ts)
            cand.extend(map(lookup.__getitem__, candidates.split()))
            offsets.append(len(cand))
            if count % 500000 == 0:
                print(split, count, flush=True)
    c = np.frombuffer(cand, dtype=np.uint32)
    if split != "test":
        save(out, "label", c & 1, "uint8")
        np.right_shift(c, 1, out=c)
    save(out, "candidate", c, "int32")
    for name, values in (("offset",offsets), ("imp_id",imp_ids), ("time",times), ("user",imp_user),
                         ("user_id",user_ids), ("hist",hist), ("hist_offset",hist_off)):
        save(out, name, values, "int32")
    meta = dict(split=split, impressions=len(imp_ids), candidates=len(c), users=len(users),
                history_rows=len(user_ids), extra_histories=hist_variant_count,
                max_candidates=int(np.diff(np.asarray(offsets)).max()),
                time_basis="naive MIND wall time expressed as seconds from 1970-01-01", source_sha256=digest.hexdigest())
    (out/"meta.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta), flush=True)

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("task", choices=("news", *SPLITS))
    task=parser.parse_args().task
    if task=="news": encode_news()
    else: encode_split(task)

