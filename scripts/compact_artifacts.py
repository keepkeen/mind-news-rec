"""Losslessly compact generated ranking inputs with APFS compression or portable Zstandard archives."""
import hashlib
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from colab_jobs import ART,atomic_json

def digest(path):
    with path.open("rb") as f:return hashlib.file_digest(f,"sha256").hexdigest()
def compress(p,marker):
    if not p.is_file():return
    st=p.stat()
    if getattr(st,"st_flags",0)&0x20 or st.st_size<24*1024**2:return
    temp=p.with_name("."+p.name+".compressing")
    if shutil.disk_usage(ART).free<st.st_size+500*1024**2:return
    try:
        subprocess.run(["/usr/bin/ditto","--noclone","--hfsCompression",str(p),str(temp)],check=True)
        t=temp.stat();saved=(st.st_blocks-t.st_blocks)*512
        if saved<st.st_size*.045:return
        want=marker.read_text().strip() if marker else digest(p)
        if st.st_size!=t.st_size or digest(temp)!=want:raise ValueError(f"Compression verification failed for {p}")
        if p.stat().st_ino!=st.st_ino or p.stat().st_mtime_ns!=st.st_mtime_ns:return
        os.replace(temp,p)
        return dict(file=str(p),logical_bytes=st.st_size,allocated_bytes=t.st_blocks*512,saved_bytes=saved,sha256=want)
    finally:temp.unlink(missing_ok=True)
def main():
    root=ART/"round6";root.mkdir(exist_ok=True,parents=True)
    out=root/"disk_receipts.json"
    log=json.loads(out.read_text()) if out.exists() else {}
    immutable=[(p,None) for p in sorted((ART/"exposure").glob("*.npy"))+sorted((ART/"compact").glob("*/*.npy"))]
    candidates=[(m.with_name(m.name.removesuffix(".verified")),m) for m in (ART/"jobs").glob("mind*/*.verified")]
    for p,marker in immutable+sorted(candidates,key=lambda x:x[0].stat().st_size if x[0].exists() else 0,reverse=True):
        if str(p) in log or not p.is_file():continue
        if marker and shutil.disk_usage(ART).free>10*1024**3:break
        receipt=compress(p,marker)
        if receipt:log[str(p)]=receipt;atomic_json(out,log)
    atomic_json(root/"disk_state.json",dict(utc=time.time(),available_bytes=shutil.disk_usage(ART).free,
                logical_files_kept=len(log),saved_bytes=sum(x["saved_bytes"] for x in log.values())))
def indexes(restore):
    import numpy as np
    import zstandard
    from exposures import INDEX,Reader,unpacked_index
    report=ART/"round6/exposure_storage.json"
    receipts=json.loads(report.read_text()) if report.exists() else {}
    def feature_digests():
        result={}
        for split in ("dev","test"):
            r=Reader(split);rows=np.linspace(0,len(r.u)-1,2048,dtype=np.intp)
            result[split]=hashlib.sha256(r.prefix(rows).tobytes()).hexdigest()
        return result
    before=feature_digests()
    for name in ("article","user_article"):
        raw=INDEX/(name+".npy");packed=raw.with_suffix(".npy.zst")
        if restore:
            if raw.exists():continue
            tmp=raw.with_suffix(".restoring")
            with packed.open("rb") as src,tmp.open("wb") as dst:zstandard.ZstdDecompressor().copy_stream(src,dst)
            if digest(tmp)!=receipts[name]["sha256"]:raise ValueError(name)
            os.replace(tmp,raw)
        elif raw.exists():
            original=digest(raw);size=raw.stat().st_size;tmp=packed.with_suffix(".packing")
            with raw.open("rb") as src,tmp.open("wb") as dst:
                zstandard.ZstdCompressor(level=3,threads=2,write_checksum=True).copy_stream(src,dst)
            with tmp.open("rb") as src,zstandard.ZstdDecompressor().stream_reader(src) as stream:
                if hashlib.file_digest(stream,"sha256").hexdigest()!=original:raise ValueError(name)
            os.replace(tmp,packed)
            values=unpacked_index(packed,packed.stat().st_mtime_ns);old=np.load(raw,mmap_mode="r")
            if old.shape!=values.shape or old.dtype!=values.dtype or not np.array_equal(old,values):raise ValueError(name)
            del values,old
            receipts[name]=dict(sha256=original,logical_bytes=size,compressed_bytes=packed.stat().st_size)
            atomic_json(report,receipts)
            raw.unlink()
    after=feature_digests()
    if before!=after:raise ValueError("Production prefix features changed")
    receipts.update(prefix_before=before,prefix_after=after,available_bytes=shutil.disk_usage(ART).free)
    atomic_json(report,receipts)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("mode",nargs="?",choices=["apfs","pack-index","restore-index"],default="apfs")
    mode=p.parse_args().mode
    if mode=="apfs":main()
    else:indexes(mode=="restore-index")
