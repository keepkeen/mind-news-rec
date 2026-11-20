"""Independently validate the final rank archive against the original unlabelled TSV."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path
import re
import zipfile

RANKS=re.compile(rb"\[[1-9][0-9]*(?:,[1-9][0-9]*)*\]")
TEST_SHAPE=(2370727,93115001)
def verify(archive,raw,*,expected_shape=TEST_SHAPE):
    lines=items=0;payload=hashlib.sha256()
    with zipfile.ZipFile(archive) as z,raw.open("rb") as source:
        entries=z.infolist()
        if len(entries)!=1 or entries[0].filename!="prediction.txt" or entries[0].is_dir():raise ValueError("Unexpected ZIP structure")
        with z.open(entries[0]) as predictions:
            for row,pred in itertools.zip_longest(source,predictions):
                if row is None or pred is None:raise ValueError(f"Length mismatch at line {lines+1}")
                payload.update(pred)
                imp,sep,slate=pred.rstrip(b"\r\n").partition(b" ")
                expected=row.partition(b"\t")[0]
                if not sep or imp!=expected or not RANKS.fullmatch(slate):raise ValueError(f"ID or rank syntax on line {lines+1}")
                n=len(row.rpartition(b"\t")[2].split())
                values=list(map(int,slate[1:-1].split(b",")))
                if len(values)!=n or min(values)!=1 or max(values)!=n or len(set(values))!=n:
                    raise ValueError(f"Not a 1..N permutation on line {lines+1}")
                lines+=1;items+=n
        crc=entries[0].CRC
    if (lines,items)!=expected_shape:raise ValueError((lines,items))
    with archive.open("rb") as f:sha=hashlib.file_digest(f,"sha256").hexdigest()
    return dict(zip=str(archive),rows=lines,candidates=items,sha256=sha,prediction_sha256=payload.hexdigest(),
        crc32=f"{crc:08x}",root_entries=["prediction.txt"],raw=str(raw),id_order=True,two_fields=True,permutations=True)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("archive",type=Path);p.add_argument("--raw",type=Path,default=Path("data/large_test/MINDlarge_test/behaviors.tsv"));p.add_argument("--out",type=Path)
    a=p.parse_args();result=verify(a.archive,a.raw)
    if a.out:
        a.out.parent.mkdir(parents=True,exist_ok=True);tmp=a.out.with_suffix(".tmp");tmp.write_text(json.dumps(result,indent=2));tmp.replace(a.out)
    print(json.dumps(result))
