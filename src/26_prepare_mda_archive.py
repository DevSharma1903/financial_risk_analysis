"""Prepare a compact, label-free MD&A text archive for Colab experiments.

The source may contain the legacy amendment-proxy column. This script never
loads it and writes only filing metadata plus truncated MD&A text.
"""
from __future__ import annotations

from pathlib import Path
import argparse
import csv
import gzip
import hashlib
import json
import re
import sys
from io import TextIOWrapper

import pandas as pd


def _sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _file_date(values: pd.Series, column: str) -> pd.Series:
    if column == "filed":
        raw=values.fillna("").astype(str).str.replace(r"\.0$","",regex=True).str.zfill(8)
        dates=pd.to_datetime(raw,format="%Y%m%d",errors="coerce")
    else:
        dates=pd.to_datetime(values,errors="coerce")
    return dates


def prepare(root: Path, source: Path | None = None, output: Path | None = None,
            proxy_path: Path | None = None, min_words: int = 200,
            max_words: int = 20000) -> dict:
    root=Path(root)
    data=root/"data"
    if source is None:
        raw=data/"mda_corpus.csv"
        prepared=data/"colab_mda_text_case_control.csv.gz"
        source=raw if raw.is_file() else prepared
    source=Path(source)
    output=Path(output) if output is not None else data/"colab_mda_archive_unlabeled.csv.gz"
    output.parent.mkdir(parents=True,exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(f"Missing MD&A source corpus: {source}")
    header=set(pd.read_csv(source,compression="infer",nrows=0).columns)
    date_column="filed" if "filed" in header else "filed_date" if "filed_date" in header else None
    required={"adsh","cik","form","mda_text"}
    if date_column is None or required-header:
        raise ValueError(f"Source corpus needs {sorted(required)} and filed or filed_date; got {sorted(header)}")
    columns=["adsh","cik","form",date_column,"mda_text"]
    reader=pd.read_csv(source,compression="infer",usecols=columns,
                       dtype={"adsh":str,"cik":str,"form":str},
                       chunksize=500,low_memory=False)
    known=set()
    rows=0; skipped_short=0; years={}; forms={}; words_total=0
    proxy_path=Path(proxy_path) if proxy_path is not None else data/"colab_amendment_proxy.csv.gz"
    proxy=None
    if proxy_path.is_file():
        proxy=pd.read_csv(proxy_path,compression="infer",
            usecols=["adsh","cik","form","filed_date"],
            dtype={"adsh":str,"cik":str,"form":str},low_memory=False)
        if proxy.adsh.isna().any() or proxy.adsh.duplicated().any():
            raise ValueError("Prepared proxy must have one unique accession per row")
        proxy=proxy.set_index("adsh")

    temporary=output.with_name(output.name+".part")
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(fileobj=raw,mode="wb",compresslevel=6,mtime=0) as compressed:
                with TextIOWrapper(compressed,encoding="utf-8",newline="") as text_file:
                    writer=csv.writer(text_file)
                    writer.writerow(["adsh","cik","form","filed_date","mda_text","mda_word_count"])
                    for chunk in reader:
                        chunk["adsh"]=chunk.adsh.fillna("").astype(str).str.strip()
                        if chunk.adsh.eq("").any() or chunk.adsh.isin(known).any():
                            raise ValueError("Source corpus has blank or duplicate accession IDs")
                        known.update(chunk.adsh.tolist())
                        chunk["form"]=chunk.form.fillna("").astype(str).str.strip().str.upper()
                        if not chunk.form.isin(["10-K","10-Q"]).all():
                            raise ValueError("Archive inputs must be original 10-K/10-Q filings")
                        dates=_file_date(chunk[date_column],date_column)
                        if dates.isna().any():
                            raise ValueError("Source corpus contains unparseable filing dates")
                        chunk["mda_text"]=chunk.mda_text.fillna("").astype(str)
                        # Bound very long records while retaining the opening and
                        # substantive MD&A text needed for compact transfer.
                        words=chunk.mda_text.str.split()
                        counts=words.str.len()
                        keep=counts.ge(min_words)
                        skipped_short+=int((~keep).sum())
                        if proxy is not None:
                            check=chunk.loc[keep,["adsh","cik","form"]].copy()
                            check["filed_date"]=dates.loc[keep].dt.strftime("%Y-%m-%d").to_numpy()
                            joined=check.merge(proxy.reset_index(),on="adsh",how="left",
                                validate="one_to_one",suffixes=("_archive","_proxy"))
                            if joined.cik_proxy.isna().any():
                                raise ValueError("Some source archive accessions are absent from the prepared feature table")
                            if (joined.form_archive.astype(str).ne(joined.form_proxy.astype(str)).any() or
                                joined.filed_date_archive.ne(pd.to_datetime(joined.filed_date_proxy,errors="coerce")
                                                             .dt.strftime("%Y-%m-%d")).any()):
                                raise ValueError("Archive filing form/date disagrees with the prepared feature table")
                        selected=chunk.loc[keep].copy()
                        selected["filed_date"]=dates.loc[keep].dt.strftime("%Y-%m-%d").to_numpy()
                        selected["mda_word_count"]=counts.loc[keep].clip(upper=max_words).astype("int32").to_numpy()
                        selected["mda_text"]=[" ".join(tokens[:max_words]) for tokens in words.loc[keep]]
                        selected["cik"]=selected.cik.fillna("").astype(str).str.lstrip("0").replace("","0")
                        for year,count in pd.to_datetime(selected.filed_date).dt.year.value_counts().items():
                            years[int(year)]=years.get(int(year),0)+int(count)
                        for form,count in selected.form.value_counts().items():
                            forms[str(form)]=forms.get(str(form),0)+int(count)
                        words_total+=int(selected.mda_word_count.sum())
                        writer.writerows(selected[["adsh","cik","form","filed_date","mda_text",
                                                   "mda_word_count"]].itertuples(index=False,name=None))
                        rows+=len(selected)
        temporary.replace(output)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    manifest={"source":str(source),"source_sha256":_sha256(source),
        "source_columns_used":columns,"output":str(output),"output_sha256":_sha256(output),
        "output_bytes":output.stat().st_size,"usable_rows":rows,
        "excluded_under_min_words":skipped_short,"min_words":min_words,
        "max_words_per_document":max_words,"total_retained_words":words_total,
        "year_counts":dict(sorted(years.items())),"form_counts":forms,
        "proxy_metadata_validated":proxy is not None,"legacy_proxy_label_read":False,
        "outcome_labels_in_output":False,
        "note":"Text-only archive for unsupervised feature learning. It does not supply restatement outcomes."}
    manifest_path=output.with_name("mda_archive_manifest.json")
    manifest_path.write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    print(json.dumps(manifest,indent=2))
    return manifest


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--source",type=Path)
    parser.add_argument("--output",type=Path)
    parser.add_argument("--proxy",type=Path)
    parser.add_argument("--min-words",type=int,default=200)
    parser.add_argument("--max-words",type=int,default=20000)
    args=parser.parse_args()
    prepare(args.root,args.source,args.output,args.proxy,args.min_words,args.max_words)
