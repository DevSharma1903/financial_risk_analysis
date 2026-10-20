"""Collect full MD&A sections from the public SEC archive for the GPU study.

This module only prepares data. It never fits a model and never changes labels.
The source manifest is the existing, fixed text manifest; all outcomes are
rejoined from the current schema-v4 dataset by accession number.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import re
import sqlite3
import sys
from pathlib import Path
from io import TextIOWrapper

import pandas as pd
from bs4 import BeautifulSoup

SRC = Path(__file__).resolve().parent
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import research_v5_core as core
import sec_v5_evidence as evidence

MAX_SECTION_WORDS = 6000
MIN_SECTION_WORDS = 200


def _item_pattern(form: str, start: bool) -> re.Pattern[str]:
    if form == '10-K':
        item = r'7' if start else r'(?:7\s*A|8)'
    elif form == '10-Q':
        item = r'2' if start else r'(?:3|4)'
    else:
        raise ValueError('Only original 10-K and 10-Q filings have supported MD&A items.')
    return re.compile(r'^\s*(?:PART\s+[IVX]+\s+)?ITEM\s+' + item + r'(?![A-Z0-9])', re.I)


def extract_mda_section(html: str, form: str, max_words: int = MAX_SECTION_WORDS) -> tuple[str, str, int]:
    """Extract the longest well-bounded Item 7/Item 2 section.

    SEC filings often repeat section labels in a table of contents. Candidate
    start/end pairs are scored by body length, which generally selects the real
    section. Ambiguous or short documents are marked unavailable instead of
    silently substituting the whole filing.
    """
    if form not in {'10-K', '10-Q'}:
        raise ValueError('Only original 10-K and 10-Q filings are accepted.')
    if max_words < MIN_SECTION_WORDS:
        raise ValueError(f'max_words must be at least {MIN_SECTION_WORDS}.')
    soup = BeautifulSoup(html or '', 'html.parser')
    for tag in soup.find_all(['script', 'style', 'ix:header', 'ix:hidden']):
        tag.decompose()
    lines = [re.sub(r'\s+', ' ', line).strip() for line in soup.get_text('\n').splitlines()]
    lines = [line for line in lines if line]
    if not lines:
        return '', 'no_text', 0

    starts = _item_pattern(form, True)
    ends = _item_pattern(form, False)
    start_indices = [i for i, line in enumerate(lines) if starts.search(line)]
    candidates = []
    for start in start_indices:
        end = next((i for i in range(start + 1, len(lines)) if ends.search(lines[i])), None)
        if end is None:
            continue
        body = '\n'.join(lines[start + 1:end]).strip()
        words = body.split()
        if len(words) >= MIN_SECTION_WORDS:
            candidates.append((len(words), body))
    if not candidates:
        return '', 'no_mda_section', 0

    word_count, body = max(candidates, key=lambda item: item[0])
    if word_count > max_words:
        body = ' '.join(body.split()[:max_words])
        return body, 'ok_truncated', max_words
    return body, 'ok', word_count


def build_manifest(root: Path) -> pd.DataFrame:
    """Return the fixed 111k-style study manifest with current proxy labels."""
    root = Path(root)
    proxy_path = root / 'data' / 'colab_amendment_proxy.csv.gz'
    if not proxy_path.is_file():
        raise FileNotFoundError(f'Missing prepared proxy dataset: {proxy_path}')
    cache_dir = root / 'evidence_v5' / 'mda_sections'
    manifest_path = cache_dir / 'text_manifest.csv.gz'
    metadata_path = cache_dir / 'text_manifest.json'
    dataset_hash = core.file_hash(proxy_path)
    if manifest_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('proxy_sha256') == dataset_hash:
            manifest = pd.read_csv(manifest_path, compression='gzip',
                                   dtype={'adsh':str,'cik':str,'form':str,'y_proxy':'int8'})
            manifest['filed_date'] = pd.to_datetime(manifest.filed_date, errors='raise')
            return manifest
    proxy = core.load_data(proxy_path)
    manifest = core.text_manifest(proxy)
    labels = proxy[['adsh', 'cik', 'form', 'filed_date', core.TARGET]].copy()
    manifest = manifest[['adsh']].merge(labels, on='adsh', how='left', validate='one_to_one')
    if manifest[core.TARGET].isna().any() or manifest.cik.isna().any():
        raise ValueError('A selected accession has no current proxy label or issuer ID.')
    manifest = manifest.rename(columns={core.TARGET: 'y_proxy'})
    manifest['cik'] = manifest.cik.astype(str).str.lstrip('0').replace('', '0')
    manifest['filed_date'] = pd.to_datetime(manifest.filed_date, errors='raise')
    if manifest.adsh.duplicated().any():
        raise ValueError('The text manifest must contain unique accessions.')
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix('.part.csv.gz')
    manifest.to_csv(temporary, index=False, compression={'method':'gzip','compresslevel':6,'mtime':0})
    temporary.replace(manifest_path)
    metadata_path.write_text(json.dumps({'proxy_sha256':dataset_hash,'rows':len(manifest)},indent=2),
                             encoding='utf-8')
    return manifest


def open_section_db(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute('''CREATE TABLE IF NOT EXISTS mda_sections (
        adsh TEXT PRIMARY KEY, cik TEXT NOT NULL, form TEXT NOT NULL,
        filed_date TEXT NOT NULL, status TEXT NOT NULL, body BLOB NOT NULL,
        word_count INTEGER NOT NULL, sha256 TEXT NOT NULL)''')
    db.commit()
    return db


def collect_sections(root: Path, user_agent: str, max_new: int = 3000) -> dict:
    """Fetch/resume only selected original filings, respecting the SEC client."""
    if max_new < 0:
        raise ValueError('max_new must be zero (all remaining) or a positive integer.')
    root = Path(root)
    manifest = build_manifest(root)
    db_path = root / 'evidence_v5' / 'mda_sections' / 'sections.sqlite'
    db = open_section_db(db_path)
    done = {row[0] for row in db.execute('SELECT adsh FROM mda_sections')}
    pending = manifest.loc[~manifest.adsh.isin(done)]
    client = evidence.SecClient(user_agent)
    attempted = 0
    try:
        for row in pending.itertuples(index=False):
            if max_new and attempted >= max_new:
                break
            attempted += 1
            adsh = str(row.adsh)
            cik = str(int(row.cik))
            accession_path = adsh.replace('-', '')
            url = f'https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{adsh}.txt'
            status = 'http_404'
            body = ''
            word_count = 0
            digest = ''
            try:
                submission = client.get(url, limit=25 * 1024**2, original_form=row.form)
                document = evidence.original_document(submission, row.form)
                if document is None:
                    status = 'original_document_missing'
                else:
                    body, status, word_count = extract_mda_section(document, row.form)
                    if body:
                        digest = hashlib.sha256(body.encode('utf-8')).hexdigest()
            except Exception as exc:
                from urllib.error import HTTPError
                if isinstance(exc, HTTPError) and exc.code == 404:
                    status = 'http_404'
                elif isinstance(exc, ValueError) and 'size limit' in str(exc):
                    status = 'size_limit'
                else:
                    raise
            db.execute('INSERT INTO mda_sections VALUES (?,?,?,?,?,?,?,?)',
                       (adsh, cik, row.form, pd.Timestamp(row.filed_date).strftime('%Y-%m-%d'),
                        status, sqlite3.Binary(gzip.compress(body.encode('utf-8'))), word_count, digest))
            if attempted % 25 == 0:
                db.commit()
                print(f'SEC filing requests this session: {attempted:,}/{len(pending):,}', flush=True)
        db.commit()
    finally:
        db.commit()
        db.close()
    return coverage(root, manifest=manifest)


def coverage(root: Path, manifest: pd.DataFrame | None = None) -> dict:
    manifest = manifest if manifest is not None else build_manifest(root)
    db_path = Path(root) / 'evidence_v5' / 'mda_sections' / 'sections.sqlite'
    if not db_path.is_file():
        return {'requested': len(manifest), 'attempted': 0, 'usable': 0, 'pending': len(manifest)}
    with open_section_db(db_path) as db:
        statuses = dict(db.execute('SELECT adsh,status FROM mda_sections'))
    status = manifest.adsh.map(statuses)
    return {'requested': len(manifest), 'attempted': int(status.notna().sum()),
            'usable': int(status.isin(['ok', 'ok_truncated']).sum()),
            'pending': int(status.isna().sum()),
            'status_counts': status.fillna('pending').value_counts().sort_index().to_dict()}


def export_dataset(root: Path, output_path: Path | None = None) -> Path:
    """Export all manifest rows, including explicit missing-text statuses."""
    root = Path(root)
    manifest = build_manifest(root)
    db_path = root / 'evidence_v5' / 'mda_sections' / 'sections.sqlite'
    if not db_path.is_file():
        raise FileNotFoundError(f'No SEC text cache exists yet: {db_path}')
    with open_section_db(db_path) as db:
        statuses = dict(db.execute('SELECT adsh,status FROM mda_sections'))
        missing = int((~manifest.adsh.isin(statuses)).sum())
        if missing:
            raise RuntimeError(f'{missing:,} filings remain pending; resume collection before export.')
        rows = db.execute('SELECT adsh,status,body,word_count,sha256 FROM mda_sections')
        cached = {adsh: (status, body, words, digest)
                  for adsh, status, body, words, digest in rows}
    output_path = output_path or root / 'data' / 'colab_mda_text_expanded.csv.gz'
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('wb') as raw:
        with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=6, mtime=0) as compressed:
            with TextIOWrapper(compressed, encoding='utf-8', newline='') as text_file:
                writer = csv.writer(text_file)
                writer.writerow(['adsh', 'cik', 'form', 'filed_date', 'y_proxy', 'mda_text',
                                 'text_status', 'mda_word_count', 'text_sha256'])
                for row in manifest.itertuples(index=False):
                    status, body, words, digest = cached[row.adsh]
                    section = gzip.decompress(body).decode('utf-8') if body else ''
                    writer.writerow([row.adsh, row.cik, row.form,
                                     pd.Timestamp(row.filed_date).strftime('%Y-%m-%d'),
                                     int(row.y_proxy), section, status, int(words), digest])
    return output_path


def prepare(root: Path, user_agent: str, max_new: int = 3000) -> tuple[dict, Path | None]:
    report = collect_sections(root, user_agent, max_new=max_new)
    output = None
    if report['pending'] == 0:
        output = export_dataset(root)
        report['dataset'] = str(output)
        report['dataset_bytes'] = output.stat().st_size
    return report, output

