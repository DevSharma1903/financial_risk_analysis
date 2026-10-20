"""Prepare the existing MD&A corpus with the current proxy labels for Colab.

This is file preparation only. It fits no models and never uses the corpus's
older, balanced `y` column as the target.
"""
from pathlib import Path
import argparse
import json
import pandas as pd


def prepare(root):
    root=Path(root); data=root/'data'
    proxy_path=data/'colab_amendment_proxy.csv.gz'
    corpus_path=data/'mda_corpus.csv'
    if not proxy_path.is_file() or not corpus_path.is_file():
        raise FileNotFoundError('Expected data/colab_amendment_proxy.csv.gz and data/mda_corpus.csv')
    proxy=pd.read_csv(proxy_path,usecols=['adsh','cik','form','filed_date','y_amendment_proxy'],
                      dtype={'adsh':'string','cik':'string','form':'string'},low_memory=False)
    if proxy.adsh.duplicated().any():
        raise ValueError('Prepared proxy accession IDs are not unique.')
    corpus=pd.read_csv(corpus_path,usecols=['adsh','form','filed','mda_text','y'],
                       dtype={'adsh':'string','form':'string'},low_memory=False)
    if corpus.adsh.duplicated().any():
        raise ValueError('MD&A corpus accession IDs are not unique.')
    corpus=corpus.rename(columns={'form':'corpus_form','filed':'corpus_filed','y':'legacy_label'})
    merged=corpus.merge(proxy,on='adsh',how='left',validate='one_to_one')
    if merged.y_amendment_proxy.isna().any():
        raise ValueError(f'{int(merged.y_amendment_proxy.isna().sum())} MD&A accessions lack a current proxy label.')
    if not merged.corpus_form.eq(merged.form).all():
        raise ValueError('Filing form mismatch between MD&A and current proxy data.')
    legacy_mismatch=int(merged.legacy_label.astype(int).ne(merged.y_amendment_proxy.astype(int)).sum())
    corpus_date=pd.to_datetime(merged.corpus_filed,format='%Y%m%d',errors='coerce')
    proxy_date=pd.to_datetime(merged.filed_date,errors='coerce')
    if corpus_date.isna().any() or not corpus_date.eq(proxy_date).all():
        raise ValueError('Filing dates do not match between the MD&A corpus and proxy data.')
    if not merged.corpus_form.isin(['10-K','10-Q']).all():
        raise ValueError('The text experiment accepts original 10-K/10-Q filings only.')
    text=merged.mda_text.fillna('').astype(str).str.slice(0,20_000).str.strip()
    words=text.str.split().str.len()
    usable=words.ge(200)
    if not usable.all():
        merged=merged.loc[usable].copy(); text=text.loc[usable]; words=words.loc[usable]
    out=merged[['adsh','cik','form','filed_date','y_amendment_proxy']].copy()
    out=out.rename(columns={'y_amendment_proxy':'y_proxy'})
    out['mda_text']=text.to_numpy()
    out['mda_word_count']=words.to_numpy()
    out_path=data/'colab_mda_text_case_control.csv.gz'
    out.to_csv(out_path,index=False,compression={'method':'gzip','compresslevel':6,'mtime':0})
    report={'source_rows':len(corpus),'prepared_rows':len(out),
            'positive_count':int(out.y_proxy.sum()),'negative_count':int(len(out)-out.y_proxy.sum()),
            'legacy_label_disagreements_not_used':legacy_mismatch,
            'positive_rows_in_full_proxy':int(proxy.y_amendment_proxy.sum()),
            'text_coverage_of_full_proxy_positives':float(out.y_proxy.sum()/proxy.y_amendment_proxy.sum()),
            'by_year_form_label':out.assign(filed_year=pd.to_datetime(out.filed_date).dt.year)
              .groupby(['filed_year','form','y_proxy']).size().unstack('y_proxy',fill_value=0)
              .reset_index().to_dict('records'),
            'target':'Current FSDS same-period 365-day amendment proxy by accession; legacy corpus y excluded',
            'evaluation_limit':'Selected case-control text corpus; do not report natural-prevalence metrics from this file',
            'output':str(out_path),'output_bytes':out_path.stat().st_size}
    report_path=data/'colab_mda_text_case_control_manifest.json'
    report_path.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    return out_path,report_path


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    prepare(args.root)
