"""Public SEC evidence: complete quarterly indexes and optional ORIGINAL text.

Only used for predictor enrichment. Existing proxy outcome labels are retained.
No contact identity is stored. One sequential client, at most two requests/sec.
"""
import gzip
import hashlib
import io
import importlib.metadata
import json
import re
import sqlite3
import time
from pathlib import Path
from contextlib import closing
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from zipfile import ZipFile
import numpy as np
import pandas as pd

FORMS = {'10-K','10-Q','10-K/A','10-Q/A','8-K','8-K/A','NT 10-K','NT 10-Q'}
RISK_PATTERNS = {
    'weakness':r'\bmaterial weaknesses?\b',
    'ineffective':r'\b(?:not effective|ineffective)\b',
    'deficiency':r'\bsignificant deficienc(?:y|ies)\b',
    'going_concern':r'\bgoing concern\b',
    'restatement':r'\brestat(?:ement|ements|ed)\b',
    'related_party':r'\brelated.part(?:y|ies)\b',
    'estimate':r'\b(?:critical accounting|accounting estimates?)\b',
    'late':r'\b(?:late filing|untimely|unable to timely)\b',
    'auditor_change':r'\b(?:auditor resignation|dismissed.*?auditor)\b',
    'remediation':r'\bremediat(?:ion|e|ed)\b',
    'controls':r'\b(?:internal controls?|disclosure controls?)\b',
}


class SecClient:
    def __init__(self, user_agent):
        if not re.search(r'\S+@\S+\.\S+', user_agent) or ' ' not in user_agent.strip():
            raise ValueError('SEC_USER_AGENT must contain a real name/organization and contact email.')
        self.user_agent = user_agent.strip()
        self.previous = 0.0

    def get(self, url, limit=80*1024**2, original_form=None):
        if not url.startswith('https://www.sec.gov/Archives/'):
            raise ValueError('Only public SEC archive URLs are allowed.')
        for attempt in range(4):
            time.sleep(max(0, .5-(time.monotonic()-self.previous)))
            self.previous = time.monotonic()
            try:
                request = Request(url, headers={'User-Agent':self.user_agent,'Accept-Encoding':'identity'})
                with urlopen(request, timeout=90) as response:
                    chunks=[]; size=0
                    while True:
                        block=response.read(64*1024)
                        if not block:
                            break
                        chunks.append(block); size+=len(block)
                        if size > limit:
                            raise ValueError('SEC response exceeded the configured size limit.')
                        if original_form and b'</DOCUMENT>' in block.upper():
                            joined=b''.join(chunks)
                            if original_document(joined,original_form) is not None:
                                return joined
                    return b''.join(chunks)
            except HTTPError as exc:
                if exc.code == 403:
                    raise RuntimeError('SEC denied access (403). Stop requests; use an allowed connection later. No bypass is attempted.') from exc
                if exc.code not in (429,500,502,503,504):
                    raise
                if attempt == 3:
                    raise RuntimeError('SEC rate/server error persisted. Cached progress is retained; retry later.') from exc
                time.sleep(min(60,10*2**attempt))
            except (URLError,TimeoutError) as exc:
                if attempt == 3:
                    raise RuntimeError('Network unavailable; cached progress is retained.') from exc
                time.sleep(3*2**attempt)


def parse_master(raw):
    records=[]
    for line in raw.decode('latin-1').splitlines():
        fields=line.split('|')
        if len(fields) != 5 or not fields[0].isdigit() or fields[2] not in FORMS:
            continue
        cik,_,form,date,path=fields
        match=re.fullmatch(r'edgar/data/(\d+)/(\d{10}-\d{2}-\d{6})\.txt',path)
        if not match or int(match[1]) != int(cik):
            continue
        records.append((str(int(cik)),form,date,match[2],path))
    if not records:
        raise ValueError('SEC index did not contain expected filing records.')
    frame=pd.DataFrame(records,columns=['cik','form','filed_date','adsh','archive_path'])
    frame['filed_date']=pd.to_datetime(frame.filed_date,errors='raise')
    return frame.drop_duplicates(['cik','adsh'])


def load_sec_history(cache, client=None):
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    parts=[]
    # Predictor dates end in 2025Q3. Later index events are unnecessary.
    for year in range(2009,2026):
        for quarter in range(1,5):
            if year==2025 and quarter>3:
                break
            path=cache/f'{year}Q{quarter}.parquet'
            if not path.exists():
                if client is None:
                    raise FileNotFoundError(f'Run the evidence notebook first: missing {path.name}')
                url=f'https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{quarter}/master.zip'
                blob=client.get(url)
                with ZipFile(io.BytesIO(blob)) as z:
                    member=next((n for n in z.namelist() if n.lower()=='master.idx'),None)
                    if member is None:
                        raise ValueError('Unexpected SEC index ZIP layout.')
                    part=parse_master(z.read(member))
                temp=path.with_suffix('.part.parquet')
                part.to_parquet(temp,index=False); temp.replace(path)
                print(f'Cached {year}Q{quarter}: {len(part):,} relevant filings',flush=True)
            part=pd.read_parquet(path)
            parts.append(part)
    return pd.concat(parts,ignore_index=True).drop_duplicates(['cik','adsh'])


def original_document(blob, form):
    if form not in ('10-K','10-Q'):
        raise ValueError('Text predictors must be an original 10-K or 10-Q.')
    text=blob.decode('utf-8',errors='replace')
    for doc in re.findall(r'<DOCUMENT>(.*?)</DOCUMENT>',text,flags=re.I|re.S):
        kind=re.search(r'<TYPE>\s*([^\r\n<]+)',doc,flags=re.I)
        body=re.search(r'<TEXT>(.*?)</TEXT>',doc,flags=re.I|re.S)
        if kind and body and kind[1].strip().upper()==form:
            return body[1]
    return None


def compact_text(html):
    from bs4 import BeautifulSoup
    soup=BeautifulSoup(html,'html.parser')
    for tag in soup.find_all(['script','style','ix:header','ix:hidden']):
        tag.decompose()
    plain=re.sub(r'\s+',' ',soup.get_text(' ',strip=True))
    words=len(plain.split())
    if words<200:
        return '', {'text_available':0.0,'text_word_log':float(np.log1p(words))}
    features={'text_available':1.0,'text_word_log':float(np.log1p(words))}
    # Compact excerpts are screening windows, not certified SEC item sections.
    excerpts=[plain[:1000]]
    for name,pattern in RISK_PATTERNS.items():
        matches=list(re.finditer(pattern,plain,flags=re.I))
        features[f'text_{name}_per_1k']=len(matches)*1000/max(words,1)
        for match in matches[:2]:
            excerpts.append(plain[max(0,match.start()-250):match.end()+500])
    return '\n'.join(dict.fromkeys(excerpts))[:10000],features


def text_db(path):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    db=sqlite3.connect(path)
    db.execute('CREATE TABLE IF NOT EXISTS texts (adsh TEXT PRIMARY KEY, url TEXT, status TEXT, body BLOB, lexical TEXT, sha256 TEXT)')
    db.commit()
    return db


def collect_text(manifest, db_path, client, max_new=3000):
    db=text_db(db_path)
    done={r[0] for r in db.execute('SELECT adsh FROM texts')}
    pending=manifest[~manifest.adsh.isin(done)]
    print(f'Text cache: {len(manifest)-len(pending):,}/{len(manifest):,} attempted; '
          f'{len(pending):,} pending. This session limit: {max_new or "all"}.',flush=True)
    try:
        for i,row in enumerate(pending.itertuples(index=False),1):
            if max_new and i>max_new:
                break
            if not re.fullmatch(r'\d{10}-\d{2}-\d{6}',row.adsh):
                raise ValueError('Invalid accession in text manifest.')
            url=f'https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/{row.adsh}.txt'
            body=''; features={'text_available':0.0}; digest=''
            try:
                blob=client.get(url,limit=25*1024**2,original_form=row.form)
                html=original_document(blob,row.form)
                if html is None:
                    status='original_document_missing'
                else:
                    body,features=compact_text(html)
                    digest=hashlib.sha256(html.encode()).hexdigest()
                    status='ok' if body else 'insufficient_text'
            except HTTPError as exc:
                if exc.code != 404:
                    raise
                status='http_404'
            except ValueError as exc:
                if 'size limit' not in str(exc):
                    raise
                status='size_limit'
            db.execute('INSERT OR REPLACE INTO texts VALUES (?,?,?,?,?,?)',
                       (row.adsh,url,status,gzip.compress(body.encode()),json.dumps(features),digest))
            if i%25==0:
                db.commit(); print(f'Text requests completed this session: {i:,}',flush=True)
    finally:
        db.commit(); db.close()
    return text_coverage(manifest,db_path)


def text_coverage(manifest,db_path):
    with closing(text_db(db_path)) as db:
        states=dict(db.execute('SELECT adsh,status FROM texts'))
    status=manifest.adsh.map(states)
    report={'requested':len(manifest),'attempted':int(status.notna().sum()),
            'usable':int(status.eq('ok').sum()),'pending':int(status.isna().sum())}
    print('Text coverage:',report,flush=True)
    return report


def gpu_text_features(manifest,db_path,cache):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for text embedding.')
    coverage=text_coverage(manifest,db_path)
    if coverage['pending']:
        raise RuntimeError('Text download is incomplete. Resume the evidence notebook, or explicitly set USE_TEXT=False.')
    if coverage['usable']/coverage['requested'] < .90:
        raise RuntimeError('Text coverage below 90%; inspect failures before enabling the text experiment.')
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info
    cache=Path(cache); cache.mkdir(parents=True,exist_ok=True)
    meta_path=cache/'encoder.json'
    name='sentence-transformers/all-MiniLM-L6-v2'
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
    else:
        meta={'model':name,'revision':model_info(name).sha,'license':'Apache-2.0',
              'source':f'https://huggingface.co/{name}','text_extraction_version':1}
        meta_path.write_text(json.dumps(meta,indent=2))
    encoder=SentenceTransformer(meta['model'],revision=meta['revision'],device='cuda',trust_remote_code=False)
    frames=[]
    with closing(text_db(db_path)) as db:
        for start in range(0,len(manifest),1000):
            ids=manifest.adsh.iloc[start:start+1000].tolist()
            raw=[db.execute('SELECT body,lexical,sha256 FROM texts WHERE adsh=?',(a,)).fetchone() for a in ids]
            packages={n:importlib.metadata.version(n) for n in
                      ('torch','sentence-transformers','transformers')}
            key=hashlib.sha256(json.dumps([ids,[r[2] for r in raw],meta,packages],sort_keys=True).encode()).hexdigest()[:24]
            path=cache/f'{key}.parquet'
            if path.exists():
                frames.append(pd.read_parquet(path)); continue
            pieces=[]; owners=[]; lex=[]
            for i,(body,features,_) in enumerate(raw):
                lex.append(json.loads(features)); words=gzip.decompress(body).decode().split()
                for offset in range(0,min(len(words),8*140),140):
                    pieces.append(' '.join(words[offset:offset+140])); owners.append(i)
            vectors=np.zeros((len(ids),384),dtype=np.float32); counts=np.zeros(len(ids))
            if pieces:
                encoded=encoder.encode(pieces,batch_size=64,show_progress_bar=True,
                                       convert_to_numpy=True,normalize_embeddings=True)
                np.add.at(vectors,np.asarray(owners),encoded)
                np.add.at(counts,np.asarray(owners),1)
                vectors/=np.maximum(counts[:,None],1)
            frame=pd.concat([pd.DataFrame({'adsh':ids}),pd.DataFrame(lex),
                             pd.DataFrame(vectors,columns=[f'text_emb_{j}' for j in range(384)])],axis=1)
            temporary=path.with_suffix('.part.parquet')
            frame.to_parquet(temporary,index=False); temporary.replace(path)
            frames.append(frame)
    del encoder
    torch.cuda.empty_cache()
    return pd.concat(frames,ignore_index=True).fillna(0)


if __name__=='__main__':
    import argparse
    import getpass
    import os
    import research_v5_core as core
    parser=argparse.ArgumentParser(description='Cache public SEC evidence without fitting any model.')
    parser.add_argument('--root',default='.')
    parser.add_argument('--text',action='store_true',help='Also download optional original-filing excerpts.')
    parser.add_argument('--max-new',type=int,default=3000,help='Text requests this session; 0 means all pending.')
    args=parser.parse_args()
    if args.max_new<0:
        parser.error('--max-new must be nonnegative')
    root=Path(args.root)
    client=SecClient(os.environ.get('SEC_USER_AGENT') or getpass.getpass('Real SEC User-Agent (name/organization and email): '))
    events=load_sec_history(root/'evidence_v5/indexes',client)
    print(f'Index cache complete: {len(events):,} relevant events.')
    if args.text:
        path=root/'data/colab_amendment_proxy.csv.gz'
        if not path.exists():
            path=root/'colab_amendment_proxy.csv.gz'
        manifest=core.text_manifest(core.load_data(path))
        manifest.to_csv(root/'evidence_v5/text_manifest.csv.gz',index=False,compression='gzip')
        collect_text(manifest,root/'evidence_v5/original_text.sqlite',client,args.max_new)
