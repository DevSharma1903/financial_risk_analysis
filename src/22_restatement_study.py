"""Prepare and evaluate a confirmed-restatement study from reviewed SEC evidence.

No outcomes are inferred from amendments. Only rows explicitly marked reviewed
in the manual ledger enter a cohort. Event dates derive 1/3/5-year targets;
negative outcomes require documented follow-up through the selected horizon.
Text downloads are limited to reviewed original filings and their prior
same-issuer/same-form filings.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import gzip
import hashlib
import importlib.metadata
import importlib.util
import json
import re
import sqlite3
import sys

import numpy as np
import pandas as pd

SRC = Path(__file__).resolve().parent
ROOT = SRC.parent
sys.path.insert(0, str(SRC)) if str(SRC) not in sys.path else None

EVENT_TYPES = {
    "restatement", "revision_not_restatement", "other_amendment",
    "no_confirmed_restatement",
}
HORIZONS = (1, 3, 5)
TRAJECTORY_RATIOS = (
    "liabilities_to_assets", "current_assets_to_current_liabilities",
    "working_capital_to_assets", "cash_to_assets", "receivables_to_revenue",
    "inventory_to_revenue", "gross_margin", "operating_margin", "net_margin",
    "net_income_to_assets", "operating_cashflow_to_assets",
    "operating_cashflow_to_net_income", "accruals_to_assets", "debt_to_assets",
    "liabilities_to_equity", "retained_earnings_to_assets",
    "capex_to_operating_cashflow", "revenue_to_assets",
)
EMBEDDING_CHUNK_WORDS = 140
LEDGER_COLUMNS = {
    "original_adsh", "original_filed_date", "label", "review_status", "event_type",
    "event_date", "outcome_followup_through", "evidence_url", "evidence_excerpt",
    "reviewer", "review_notes",
}
NON_FEATURES = {
    "adsh", "cik", "period", "period_date", "filed", "filed_date", "filed_year",
    "sic", "afs", "wksi", "days_to_amendment", "y_amendment_proxy",
    "feature_schema_version", "label_observation_cutoff", "label", "event_type",
    "event_date", "outcome_followup_through", "review_status", "evidence_url",
    "evidence_excerpt", "reviewer", "review_notes", "sampling_group",
}


def validate_ledger(frame: pd.DataFrame) -> list[str]:
    """Validate the working queue and its reviewed rows; pending rows are allowed."""
    errors = []
    missing = LEDGER_COLUMNS - set(frame.columns)
    if missing:
        return [f"Missing required ledger columns: {sorted(missing)}"]
    ids = frame.original_adsh.fillna("").astype(str).str.strip()
    if ids.eq("").any() or ids[ids.ne("")].duplicated().any():
        errors.append("Each queue row needs a unique original_adsh")
    statuses = frame.review_status.fillna("").astype(str).str.strip().str.lower()
    if (~statuses.isin(["pending", "reviewed"])).any():
        errors.append("review_status must be pending or reviewed")
    reviewed = frame.loc[statuses.eq("reviewed")].copy()
    if reviewed.empty:
        return errors
    labels = pd.to_numeric(reviewed.label, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        errors.append("Every reviewed row needs label 0 or 1")
    event_type = reviewed.event_type.fillna("").astype(str).str.strip()
    if not event_type.isin(EVENT_TYPES).all():
        errors.append(f"Reviewed event_type must be one of {sorted(EVENT_TYPES)}")
    if labels.notna().all() and not labels.astype(int).eq(event_type.eq("restatement").astype(int)).all():
        errors.append("label must be 1 exactly when event_type=restatement")
    original = pd.to_datetime(reviewed.original_filed_date, errors="coerce")
    event = pd.to_datetime(reviewed.event_date, errors="coerce")
    followup = pd.to_datetime(reviewed.outcome_followup_through, errors="coerce")
    if original.isna().any() or followup.isna().any():
        errors.append("Reviewed rows need parseable original_filed_date and outcome_followup_through")
    event_required = event_type.ne("no_confirmed_restatement")
    if (event_required & event.isna()).any():
        errors.append("A documented amendment/restatement event needs event_date")
    if ((event.notna()) & event.lt(original)).any():
        errors.append("event_date cannot precede original_filed_date")
    if ((event.notna()) & event.gt(followup)).any():
        errors.append("outcome_followup_through cannot precede event_date")
    if ((followup.notna()) & followup.lt(original)).any():
        errors.append("outcome_followup_through cannot precede original_filed_date")
    for col in ("evidence_url", "evidence_excerpt", "reviewer"):
        if reviewed[col].fillna("").astype(str).str.strip().eq("").any():
            errors.append(f"Every reviewed row needs {col}")
    urls = reviewed.evidence_url.fillna("").astype(str).str.lower()
    if (~urls.str.startswith("https://www.sec.gov/")).any():
        errors.append("evidence_url must be a primary SEC.gov filing URL")
    return errors


def derive_horizon_labels(ledger: pd.DataFrame, horizon_years: int) -> tuple[pd.DataFrame, int]:
    """Return reviewed rows observable at H years, with a horizon-specific target."""
    if horizon_years not in HORIZONS:
        raise ValueError(f"horizon_years must be one of {HORIZONS}")
    errors = validate_ledger(ledger)
    if errors:
        raise ValueError("Invalid review ledger: " + "; ".join(errors))
    reviewed = ledger.loc[ledger.review_status.fillna("").astype(str).str.lower().eq("reviewed")].copy()
    if reviewed.empty:
        return reviewed.assign(target_restatement=pd.Series(dtype="int8")), 0
    original = pd.to_datetime(reviewed.original_filed_date, errors="raise")
    end = original + pd.DateOffset(years=horizon_years)
    event = pd.to_datetime(reviewed.event_date, errors="coerce")
    followup = pd.to_datetime(reviewed.outcome_followup_through, errors="coerce")
    is_restatement = reviewed.event_type.astype(str).eq("restatement")
    positive = is_restatement & event.le(end)
    # A later documented restatement is a valid horizon-negative only if the
    # review establishes it occurred after H; other negatives need full H follow-up.
    event_later = event.notna() & event.gt(end)
    mature = positive | (event_later & followup.ge(event)) | followup.ge(end)
    ready = reviewed.loc[mature].copy()
    ready["target_restatement"] = positive.loc[mature].astype("int8").to_numpy()
    ready["horizon_end"] = end.loc[mature].dt.strftime("%Y-%m-%d").to_numpy()
    ready["outcome_followup_through"] = followup.loc[mature].dt.strftime("%Y-%m-%d").to_numpy()
    return ready.reset_index(drop=True), int((~mature).sum())


def validate_proxy_features(proxy: pd.DataFrame) -> None:
    """Require the reviewed study's known point-in-time feature schema."""
    required={"adsh","cik","form","filed_date","feature_schema_version","label_observation_cutoff"}
    missing=required-set(proxy.columns)
    if missing:
        raise ValueError(f"Prepared feature table is missing required schema-4 columns: {sorted(missing)}")
    if not pd.to_numeric(proxy.feature_schema_version,errors="coerce").eq(4).all():
        raise ValueError("Use the prepared feature_schema_version=4 table")
    if not proxy.adsh.is_unique or proxy.adsh.isna().any():
        raise ValueError("Prepared feature table needs one unique accession per row")
    if not set(proxy.form.dropna().astype(str)) <= {"10-K","10-Q"}:
        raise ValueError("Predictors must come from original 10-K/10-Q filings")
    cutoff=pd.to_datetime(proxy.label_observation_cutoff,errors="coerce")
    if cutoff.isna().any() or cutoff.nunique()!=1:
        raise ValueError("Prepared feature table needs one common parseable label observation cutoff")


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _prior_filing_links(proxy: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    """Select the immediately prior same-issuer/form filing, strictly by date."""
    rows = []
    history = proxy[["adsh", "cik", "form", "filed_date"]].copy()
    history["cik_norm"] = history.cik.astype(str).str.lstrip("0").replace("", "0")
    history["filed_date"] = pd.to_datetime(history.filed_date, errors="raise")
    history = history.sort_values(["cik_norm", "form", "filed_date", "adsh"])
    buckets = {(str(cik), str(form)): group for (cik, form), group in
               history.groupby(["cik_norm", "form"], sort=False)}
    for row in targets.itertuples(index=False):
        cik = str(row.cik).lstrip("0") or "0"
        group = buckets.get((cik, str(row.original_form)))
        prior = ""
        if group is not None:
            earlier = group.loc[group.filed_date.lt(pd.Timestamp(row.original_filed_date))]
            if not earlier.empty:
                prior = str(earlier.iloc[-1].adsh)
        rows.append((str(row.original_adsh), prior))
    return pd.DataFrame(rows, columns=["original_adsh", "prior_adsh"])


def _financial_trajectory_features(proxy: pd.DataFrame, targets: pd.DataFrame,
                                    max_prior_filings: int = 3) -> pd.DataFrame:
    """Build point-in-time trends from the current and up to three earlier same-form filings.

    Only a fixed set of financial ratios is read. For each target filing, every
    historical observation must have a strictly earlier filing date. This adds
    issuer-specific financial movement without using the future outcome or
    any amendment/review metadata.
    """
    if max_prior_filings < 1:
        raise ValueError("max_prior_filings must be positive")
    ratio_cols = [name for name in TRAJECTORY_RATIOS if name in proxy.columns]
    if not ratio_cols:
        return pd.DataFrame({"adsh": targets.adsh.astype(str)})
    required = {"adsh", "cik", "form", "filed_date"}
    if not required.issubset(proxy.columns) or not required.issubset(targets.columns):
        raise ValueError("Financial history needs adsh, cik, form, and filed_date")

    history = proxy[["adsh", "cik", "form", "filed_date", *ratio_cols]].copy()
    history["adsh"] = history.adsh.astype(str)
    history["cik_norm"] = history.cik.astype(str).str.lstrip("0").replace("", "0")
    history["filed_date"] = pd.to_datetime(history.filed_date, errors="coerce")
    if history.filed_date.isna().any() or history.adsh.duplicated().any():
        raise ValueError("Financial history needs unique accessions and parseable filing dates")
    for name in ratio_cols:
        history[name] = pd.to_numeric(history[name], errors="coerce").replace(
            [np.inf, -np.inf], np.nan)
    history = history.sort_values(["cik_norm", "form", "filed_date", "adsh"])
    groups = {(str(cik), str(form)): group for (cik, form), group in
              history.groupby(["cik_norm", "form"], sort=False)}

    result = []
    for target in targets.itertuples(index=False):
        cik = str(target.cik).lstrip("0") or "0"
        filed = pd.Timestamp(target.filed_date)
        group = groups.get((cik, str(target.form)))
        earlier = (group.loc[group.filed_date.lt(filed)].tail(max_prior_filings)
                   if group is not None else history.iloc[:0])
        current = {name: pd.to_numeric(getattr(target, name, np.nan), errors="coerce")
                   for name in ratio_cols}
        record = {"adsh": str(target.adsh),
                  "trajectory_prior_filing_count": int(len(earlier))}
        if len(earlier):
            record["trajectory_days_since_prior_filing"] = int(
                (filed - earlier.filed_date.iloc[-1]).days)
        else:
            record["trajectory_days_since_prior_filing"] = np.nan
        for name in ratio_cols:
            prior_values = earlier[name].to_numpy(dtype=float) if len(earlier) else np.array([])
            current_value = current[name]
            values = np.r_[prior_values, current_value]
            valid = np.isfinite(values)
            x = np.arange(len(values), dtype=float)[valid]
            y = values[valid]
            prefix = f"trajectory__{name}__"
            record[prefix + "slope_per_filing"] = (
                float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else np.nan)
            record[prefix + "std"] = float(np.std(y)) if len(y) >= 2 else np.nan
            record[prefix + "range"] = float(np.ptp(y)) if len(y) >= 2 else np.nan
            finite_prior = prior_values[np.isfinite(prior_values)]
            record[prefix + "latest_step"] = (
                float(current_value - finite_prior[-1])
                if np.isfinite(current_value) and len(finite_prior) else np.nan)
        result.append(record)
    return pd.DataFrame(result)


def section_db(path: Path) -> sqlite3.Connection:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(path)
    db.execute("""CREATE TABLE IF NOT EXISTS restatement_mda (
        adsh TEXT PRIMARY KEY, cik TEXT NOT NULL, form TEXT NOT NULL,
        filed_date TEXT NOT NULL, status TEXT NOT NULL, body BLOB NOT NULL,
        word_count INTEGER NOT NULL, sha256 TEXT NOT NULL)""")
    db.commit()
    return db


def collect_reviewed_mda(proxy: pd.DataFrame, ledger: pd.DataFrame, horizon_years: int,
                         db_path: Path, user_agent: str, max_new: int = 500,
                         client=None) -> dict:
    """Fetch only reviewed target originals and their prior comparable originals."""
    ready, immature = derive_horizon_labels(ledger, horizon_years)
    if ready.empty:
        raise ValueError("No reviewed rows have an observable outcome at this horizon.")
    proxy = proxy.copy()
    validate_proxy_features(proxy)
    proxy["filed_date"] = pd.to_datetime(proxy.filed_date, errors="raise")
    proxy["cik"] = proxy.cik.astype(str).str.lstrip("0").replace("", "0")
    if not proxy.adsh.is_unique:
        raise ValueError("Prepared feature table must have unique accession IDs")
    target_ids = set(ready.original_adsh.astype(str))
    targets = proxy.loc[proxy.adsh.astype(str).isin(target_ids),
                        ["adsh", "cik", "form", "filed_date"]].rename(columns={
                            "adsh":"original_adsh", "form":"original_form",
                            "filed_date":"proxy_filed_date"})
    if len(targets) != len(ready):
        missing = target_ids - set(targets.original_adsh.astype(str))
        raise ValueError(f"{len(missing)} reviewed accessions do not match the prepared feature table; first={sorted(missing)[:3]}")
    targets = ready[["original_adsh", "original_filed_date"]].merge(
        targets, on="original_adsh", validate="one_to_one")
    targets["original_filed_date"] = pd.to_datetime(targets.original_filed_date, errors="raise")
    targets["proxy_filed_date"] = pd.to_datetime(targets.proxy_filed_date, errors="raise")
    if not targets.original_filed_date.eq(targets.proxy_filed_date).all():
        raise ValueError("The reviewed original_filed_date does not match the prepared accession record")
    if not targets.original_form.isin(["10-K", "10-Q"]).all():
        raise ValueError("Only original 10-K and 10-Q filings may supply predictors")
    links = _prior_filing_links(proxy, targets)
    target_manifest = targets.rename(columns={"original_adsh":"adsh"})[
        ["adsh", "cik", "original_form", "original_filed_date"]].rename(columns={
            "original_form":"form", "original_filed_date":"filed_date"})
    prior_ids = set(links.prior_adsh) - {""}
    prior_manifest = proxy.loc[proxy.adsh.astype(str).isin(prior_ids),
        ["adsh", "cik", "form", "filed_date"]].copy()
    manifest = pd.concat([target_manifest, prior_manifest], ignore_index=True).drop_duplicates("adsh")
    manifest["adsh"] = manifest.adsh.astype(str)
    manifest["cik"] = manifest.cik.astype(str).str.lstrip("0").replace("", "0")
    manifest["filed_date"] = pd.to_datetime(manifest.filed_date, errors="raise")

    evidence = _load_module("restatement_sec_evidence", SRC / "sec_v5_evidence.py")
    advanced = _load_module("restatement_mda_extractor", SRC / "19_mda_advanced.py")
    client = client or evidence.SecClient(user_agent)
    db = section_db(db_path)
    done = {r[0] for r in db.execute("SELECT adsh FROM restatement_mda")}
    pending = manifest.loc[~manifest.adsh.isin(done)]
    attempted = 0
    try:
        for row in pending.itertuples(index=False):
            if max_new and attempted >= max_new:
                break
            attempted += 1
            cik = str(int(row.cik))
            accession_path = str(row.adsh).replace("-", "")
            url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_path}/{row.adsh}.txt"
            status, body, words, digest = "", "", 0, ""
            try:
                blob = client.get(url, limit=80 * 1024**2, original_form=row.form)
                html = evidence.original_document(blob, row.form)
                if html is None:
                    status = "original_document_missing"
                else:
                    body, status, words = advanced.extract_mda_section(html, row.form)
                    digest = hashlib.sha256(html.encode("utf-8", errors="replace")).hexdigest()
            except Exception as exc:
                # Access denial, rate limits, and network errors remain visible
                # and stop the run; missing/oversize documents are cached below.
                from urllib.error import HTTPError
                if isinstance(exc, HTTPError) and exc.code == 404:
                    status = "http_404"
                elif isinstance(exc, ValueError) and "size limit" in str(exc):
                    status = "size_limit"
                else:
                    raise
            db.execute("INSERT OR REPLACE INTO restatement_mda VALUES (?,?,?,?,?,?,?,?)",
                       (row.adsh, cik, row.form, pd.Timestamp(row.filed_date).strftime("%Y-%m-%d"),
                        status, gzip.compress(body.encode("utf-8")), int(words), digest))
            if attempted % 25 == 0:
                db.commit()
                print(f"SEC filing texts cached: {attempted:,}/{len(pending):,} in this session", flush=True)
    finally:
        db.commit()
        db.close()
    with closing(section_db(db_path)) as db:
        states = dict(db.execute("SELECT adsh,status FROM restatement_mda"))
    counts = pd.Series([states.get(a, "pending") for a in manifest.adsh]).value_counts().to_dict()
    return {"horizon_years":horizon_years, "reviewed_mature_rows":len(ready),
            "excluded_immature_rows":immature, "target_rows":len(targets),
            "prior_filings":len(prior_ids), "unique_filings":len(manifest),
            "attempted_this_session":attempted, "status_counts":counts,
            "pending":int(sum(v for k,v in counts.items() if k == "pending"))}


def prepare_model_data(proxy_path: Path, ledger_path: Path, sections_db_path: Path,
                       horizon_years: int, output_path: Path) -> dict:
    """Join reviewed horizon labels, point-in-time financials, and cached MD&A."""
    ledger = pd.read_csv(ledger_path, dtype={"original_adsh":str}, low_memory=False)
    ready, immature = derive_horizon_labels(ledger, horizon_years)
    if ready.empty:
        raise ValueError("No reviewed outcomes have sufficient follow-up; review labels first.")
    proxy = pd.read_csv(proxy_path, dtype={"adsh":str,"cik":str}, low_memory=False)
    validate_proxy_features(proxy)
    if not proxy.adsh.is_unique:
        raise ValueError("Prepared feature table contains duplicate accessions")
    ids = set(ready.original_adsh.astype(str))
    part = proxy.loc[proxy.adsh.astype(str).isin(ids)].copy()
    if len(part) != len(ready):
        raise ValueError("Some reviewed accessions are absent from the prepared feature table")
    part = part.merge(ready[["original_adsh", "target_restatement", "horizon_end",
                             "outcome_followup_through"]].rename(columns={"original_adsh":"adsh"}),
                      on="adsh", validate="one_to_one")
    trajectory = _financial_trajectory_features(proxy, part)
    part = part.merge(trajectory, on="adsh", validate="one_to_one")
    part["original_adsh"] = part.adsh.astype(str)
    part["filed_date"] = pd.to_datetime(part.filed_date, errors="raise")
    link_proxy = proxy[["adsh", "cik", "form", "filed_date"]].copy()
    link_proxy["cik"] = link_proxy.cik.astype(str).str.lstrip("0").replace("", "0")
    targets = part[["adsh", "cik", "form", "filed_date"]].rename(columns={
        "adsh":"original_adsh", "form":"original_form", "filed_date":"original_filed_date"})
    targets["cik"] = targets.cik.astype(str).str.lstrip("0").replace("", "0")
    links = _prior_filing_links(link_proxy, targets)
    part = part.merge(links, on="original_adsh", validate="one_to_one")
    with closing(section_db(sections_db_path)) as db:
        texts = {a:(gzip.decompress(body).decode("utf-8", errors="replace"), status)
                 for a,body,status in db.execute("SELECT adsh,body,status FROM restatement_mda")}
    part["mda_text"] = part.adsh.astype(str).map(lambda a:texts.get(a,("","missing"))[0])
    part["mda_status"] = part.adsh.astype(str).map(lambda a:texts.get(a,("","pending"))[1])
    part["prior_mda_text"] = part.prior_adsh.astype(str).map(lambda a:texts.get(a,("","missing"))[0])
    part["prior_mda_status"] = part.prior_adsh.astype(str).map(lambda a:texts.get(a,("","pending"))[1])
    # Never pass label, future outcome, accession, issuer ID, dates, or the
    # existing amendment proxy target to predictors. Past issuer histories and
    # financial deltas remain legitimate as-filed features.
    keep_meta = {"adsh", "cik", "form", "fp", "sic2", "filed_date", "target_restatement",
                 "horizon_end", "outcome_followup_through", "prior_adsh", "mda_text",
                 "mda_status", "prior_mda_text", "prior_mda_status"}
    feature_cols = [c for c in part if c not in NON_FEATURES and c not in keep_meta and
                    not c.startswith(("label_", "target_", "evidence_", "event_"))]
    meta_order = [c for c in ["adsh", "cik", "form", "fp", "sic2", "filed_date",
        "target_restatement", "horizon_end", "outcome_followup_through", "prior_adsh",
        "mda_text", "mda_status", "prior_mda_text", "prior_mda_status"] if c in part]
    safe = part[meta_order + feature_cols].copy()
    if safe.adsh.duplicated().any():
        raise ValueError("Prepared model table must have one row per original filing")
    safe["filed_year"] = pd.to_datetime(safe.filed_date).dt.year.astype("int16")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp = output_path.with_name(output_path.name + ".part")
    safe.to_csv(temp, index=False, compression="gzip")
    temp.replace(output_path)
    return {"output":str(output_path), "rows":len(safe),
            "positives":int(safe.target_restatement.sum()),
            "negatives":int((safe.target_restatement == 0).sum()),
            "excluded_immature":immature,
            "current_mda_usable":int(safe.mda_status.isin(["ok", "ok_truncated"]).sum()),
            "prior_mda_usable":int(safe.prior_mda_status.isin(["ok", "ok_truncated"]).sum()),
            "feature_count":len(feature_cols)}


def _lexical_features(text: str) -> dict[str, float]:
    text = (text or "").lower()
    words = max(1, len(re.findall(r"\b[\w'-]+\b", text)))
    cues = {
        "restatement":r"\brestat(?:e|ed|ement|ements|ing)\w*\b",
        "error":r"\b(?:error|errors|erroneous|misstatement|misstatements)\b",
        "nonreliance":r"\bnon.?reliance\b|should not be relied upon",
        "material_weakness":r"\bmaterial weaknesses?\b",
        "control_deficiency":r"\b(?:significant deficiency|ineffective internal control|control deficiencies)\b",
        "accounting_estimate":r"\b(?:accounting estimate|critical accounting|judgment|assumption)\w*\b",
        "uncertainty":r"\b(?:uncertain|uncertainty|may materially|could materially|reasonably possible)\b",
        "investigation":r"\b(?:investigation|audit committee|subpoena|inquiry)\w*\b",
        "remediation":r"\bremediat\w*\b",
        "revision":r"\b(?:revision|revised|correction|corrected|adjustment)\w*\b",
        "forward_looking":r"\b(?:expect|anticipate|believe|forecast|project|estimate)\w*\b",
    }
    out = {f"lex_{name}_per_1k":len(re.findall(pattern,text))/words*1000
           for name,pattern in cues.items()}
    out["lex_word_count_log"] = float(np.log1p(words - 1))
    return out


def _document_chunks(text: str, chunk_words: int = EMBEDDING_CHUNK_WORDS) -> list[str]:
    """Split all available MD&A text into non-overlapping encoder-sized chunks."""
    if chunk_words < 1:
        raise ValueError("chunk_words must be positive")
    words = (text or "").split()
    return [" ".join(words[start:start + chunk_words])
            for start in range(0, len(words), chunk_words)]


def _embed_sections(frame: pd.DataFrame):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. In Colab select Runtime > Change runtime type > T4 GPU.")
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import model_info
    encoder_name="sentence-transformers/all-MiniLM-L6-v2"
    encoder_revision=model_info(encoder_name).sha
    model = SentenceTransformer(encoder_name,revision=encoder_revision,device="cuda",
                                trust_remote_code=False)
    texts = pd.concat([frame.mda_text.fillna(""), frame.prior_mda_text.fillna("")],
                      ignore_index=True).astype(str).tolist()
    vecs = np.zeros((len(texts), 384), dtype=np.float32)
    chunks, owners = [], []
    for idx, text in enumerate(texts):
        # The filing extractor retains up to 6,000 words. Encode the entire
        # available section instead of silently discarding everything after
        # its first 1,120 words.
        for chunk in _document_chunks(text):
            chunks.append(chunk)
            owners.append(idx)
    if chunks:
        encoded = model.encode(chunks, batch_size=64, show_progress_bar=True,
                               convert_to_numpy=True, normalize_embeddings=True)
        counts = np.zeros(len(texts), dtype=np.float32)
        for owner, vector in zip(owners, encoded):
            vecs[owner] += vector
            counts[owner] += 1
        nonzero = counts > 0
        vecs[nonzero] /= counts[nonzero,None]
        norms = np.linalg.norm(vecs,axis=1)
        vecs[nonzero] /= np.maximum(norms[nonzero,None],1e-8)
    current, prior = vecs[:len(frame)], vecs[len(frame):]
    current_lex = pd.DataFrame([_lexical_features(t) for t in frame.mda_text.fillna("")])
    prior_lex = pd.DataFrame([_lexical_features(t) for t in frame.prior_mda_text.fillna("")])
    lexical_names = list(current_lex.columns)
    current_lex_values = current_lex.to_numpy(dtype=np.float32)
    prior_lex_values = prior_lex.to_numpy(dtype=np.float32)
    lexical_delta = current_lex_values - prior_lex_values
    delta = current - prior
    cosine = np.sum(current*prior,axis=1).reshape(-1,1)
    novelty = np.concatenate([delta, np.abs(delta), cosine],axis=1).astype(np.float32)
    status = frame.mda_status.isin(["ok","ok_truncated"]).to_numpy(dtype=np.float32).reshape(-1,1)
    prior_status = frame.prior_mda_status.isin(["ok","ok_truncated"]).to_numpy(dtype=np.float32).reshape(-1,1)
    current_embedding_names = [f"mda_current_embedding__{i:03d}" for i in range(current.shape[1])]
    prior_embedding_names = [f"mda_prior_embedding__{i:03d}" for i in range(prior.shape[1])]
    change_embedding_names = [f"mda_change_embedding__{i:03d}" for i in range(delta.shape[1])]
    abs_change_names = [f"mda_abs_change_embedding__{i:03d}" for i in range(delta.shape[1])]
    current_lex_names = [f"lex_current__{name}" for name in lexical_names]
    prior_lex_names = [f"lex_prior__{name}" for name in lexical_names]
    delta_lex_names = [f"lex_change__{name}" for name in lexical_names]
    availability_names = ["availability__current_mda", "availability__prior_mda"]
    text_names = (current_embedding_names + current_lex_names + availability_names[:1])
    novelty_names = (current_embedding_names + prior_embedding_names + change_embedding_names +
                     abs_change_names + ["mda_embedding_cosine_similarity"] + current_lex_names +
                     prior_lex_names + availability_names)
    interpretable_names = current_lex_names + prior_lex_names + delta_lex_names + availability_names
    text_view = np.concatenate([current,current_lex_values,status],axis=1)
    full_text = np.concatenate([current,prior,novelty,current_lex_values,
                                prior_lex_values],axis=1)
    interpretable = np.concatenate([current_lex_values,prior_lex_values,lexical_delta,
                                    status,prior_status],axis=1)
    return ({"text":text_view,"text_names":text_names,
             "novelty":np.concatenate([full_text,status,prior_status],axis=1),
             "novelty_names":novelty_names,
             "interpretable":interpretable,"interpretable_names":interpretable_names},
            {"name":encoder_name,"revision":encoder_revision,"license":"Apache-2.0",
             "source_section_max_words":6000,"embedding_chunk_words":EMBEDDING_CHUNK_WORDS,
             "embedding_coverage":"all chunks across the available MD&A section",
             "lexical_features":lexical_names})


def _archive_topic_features(frame: pd.DataFrame, archive_path: Path,
                             holdout_start_year: int, n_components: int = 48,
                             min_df: int = 5, max_features: int = 30000):
    """Learn an unsupervised MD&A vocabulary/topics from pre-holdout corpus text.

    The archive's legacy proxy labels are deliberately never read. Only corpus
    documents filed before the temporal test window fit TF-IDF and SVD. The
    learned transform is then applied to reviewed current/prior MD&A sections.
    """
    from sklearn.decomposition import TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer

    archive_path=Path(archive_path)
    required={"adsh","filed_date","mda_text"}
    archive=pd.read_csv(archive_path,compression="infer",usecols=sorted(required),
                        dtype={"adsh":str},low_memory=False)
    if archive.adsh.isna().any() or archive.adsh.astype(str).duplicated().any():
        raise ValueError("Full MD&A archive needs one unique, nonblank accession per row")
    archive["filed_date"]=pd.to_datetime(archive.filed_date,errors="coerce")
    if archive.filed_date.isna().any():
        raise ValueError("Full MD&A archive contains unparseable filed_date values")
    archive["mda_text"]=archive.mda_text.fillna("").astype(str)
    fit=archive.loc[archive.filed_date.dt.year.lt(int(holdout_start_year))].copy()
    fit=fit.loc[fit.mda_text.str.split().str.len().ge(200)]
    if len(fit)<max(20,min_df*2):
        raise ValueError(f"Only {len(fit)} usable pre-holdout archive documents; need a broader text archive")
    vectorizer=TfidfVectorizer(
        lowercase=True,strip_accents="unicode",sublinear_tf=True,
        ngram_range=(1,2),min_df=min_df,max_df=.995,max_features=max_features,
        dtype=np.float32,token_pattern=r"(?u)\b[\w][\w'-]*\b")
    matrix=vectorizer.fit_transform(fit.mda_text.tolist())
    components=min(int(n_components),matrix.shape[0]-1,matrix.shape[1]-1)
    if components<2:
        raise ValueError("Pre-holdout archive text has too few distinct terms for topic features")
    svd=TruncatedSVD(n_components=components,n_iter=5,random_state=20261021)
    svd.fit(matrix)

    current=frame.mda_text.fillna("").astype(str).tolist()
    prior=frame.prior_mda_text.fillna("").astype(str).tolist()
    projected=svd.transform(vectorizer.transform(current+prior)).astype(np.float32)
    current_topics=projected[:len(frame)]
    prior_topics=projected[len(frame):]
    delta=current_topics-prior_topics
    cosine=np.sum(current_topics*prior_topics,axis=1,keepdims=True)
    topics=np.concatenate([current_topics,prior_topics,delta,np.abs(delta),cosine],axis=1)
    names=([f"archive_current_topic__{i:03d}" for i in range(components)] +
           [f"archive_prior_topic__{i:03d}" for i in range(components)] +
           [f"archive_change_topic__{i:03d}" for i in range(components)] +
           [f"archive_abs_change_topic__{i:03d}" for i in range(components)] +
           ["archive_topic_cosine_similarity"])
    matched_ids=(set(frame.adsh.astype(str)) if "adsh" in frame else set())
    meta={"archive_sha256":_file_sha256(archive_path),"archive_rows":int(len(archive)),
          "archive_rows_matching_reviewed_originals":int(archive.adsh.astype(str).isin(matched_ids).sum()),
          "archive_fit_rows":int(len(fit)),"archive_rows_excluded_as_holdout_or_later":
              int(archive.filed_date.dt.year.ge(int(holdout_start_year)).sum()),
          "fit_through_year":int(holdout_start_year)-1,"holdout_start_year":int(holdout_start_year),
          "vocabulary_size":int(len(vectorizer.vocabulary_)),"topic_components":int(components),
          "legacy_proxy_labels_read":False,
          "method":"word unigram/bigram TF-IDF plus truncated SVD, fit on pre-holdout MD&A only"}
    return {"archive_topics":topics,"archive_topic_names":names},meta,vectorizer,svd


def _features(train: pd.DataFrame, valid: pd.DataFrame, view: str,
              embeddings: dict[str,np.ndarray], train_idx: np.ndarray, valid_idx: np.ndarray,
              return_names: bool = False):
    from sklearn.preprocessing import OneHotEncoder
    numeric = [c for c in train.columns if c not in NON_FEATURES and not c.startswith((
        "target_", "label_", "event_", "evidence_", "candidate_", "review_", "sampling_")) and c not in {
        "adsh","cik","form","fp","sic2","filed_date","filed_year","target_restatement",
        "horizon_end","outcome_followup_through","prior_adsh","mda_text","mda_status",
        "prior_mda_text","prior_mda_status","_embed_pos"} and pd.api.types.is_numeric_dtype(train[c])]
    categorical = [c for c in ("form","fp","sic2") if c in train]
    feature_names = ([f"structured__{c}" for c in numeric] +
                     [f"structured_missing__{c}" for c in numeric])
    def numeric_matrix(frame):
        vals = frame[numeric].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan)
        missing = vals.isna().to_numpy(dtype=np.float32)
        medians = train[numeric].apply(pd.to_numeric,errors="coerce").replace([np.inf,-np.inf],np.nan).median().fillna(0)
        # Fit imputation values only on the training fold.
        vals = vals.fillna(medians).clip(-1e12,1e12).to_numpy(dtype=np.float32)
        return np.concatenate([vals,missing],axis=1)
    numeric_train, numeric_valid = numeric_matrix(train), numeric_matrix(valid)
    if categorical:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32)
        cat_train=encoder.fit_transform(train[categorical].fillna("unknown").astype(str))
        cat_valid=encoder.transform(valid[categorical].fillna("unknown").astype(str))
        numeric_train=np.concatenate([numeric_train,cat_train],axis=1)
        numeric_valid=np.concatenate([numeric_valid,cat_valid],axis=1)
        feature_names += [f"categorical__{name}" for name in encoder.get_feature_names_out(categorical)]
    if view == "financial":
        result=(numeric_train,numeric_valid,feature_names)
    elif view == "mda":
        result=(embeddings["text"][train_idx],embeddings["text"][valid_idx],
                embeddings["text_names"])
    elif view == "interpretable":
        result=(np.concatenate([numeric_train,embeddings["interpretable"][train_idx]],axis=1),
                np.concatenate([numeric_valid,embeddings["interpretable"][valid_idx]],axis=1),
                feature_names+embeddings["interpretable_names"])
    elif view == "full_archive":
        result=(np.concatenate([numeric_train,embeddings["novelty"][train_idx],
                                embeddings["archive_topics"][train_idx]],axis=1),
                np.concatenate([numeric_valid,embeddings["novelty"][valid_idx],
                                embeddings["archive_topics"][valid_idx]],axis=1),
                feature_names+embeddings["novelty_names"]+embeddings["archive_topic_names"])
    else:
        result=(np.concatenate([numeric_train,embeddings["novelty"][train_idx]],axis=1),
                np.concatenate([numeric_valid,embeddings["novelty"][valid_idx]],axis=1),
                feature_names+embeddings["novelty_names"])
    return result if return_names else result[:2]


def _balance(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Make a transparent 1:1 year/form matched set, with no replacement."""
    parts=[]
    keys=[c for c in ("filed_year","form") if c in frame]
    if not keys:
        keys=["form"] if "form" in frame else []
    groups=[((),frame)] if not keys else frame.groupby(keys,dropna=False,sort=True)
    for _,group in groups:
        pos=group[group.target_restatement.eq(1)]
        neg=group[group.target_restatement.eq(0)]
        n=min(len(pos),len(neg))
        if n:
            parts.extend([pos.sample(n,random_state=seed),neg.sample(n,random_state=seed+1)])
    if not parts:
        return frame.iloc[:0].copy()
    return pd.concat(parts).sample(frac=1,random_state=seed+2).reset_index(drop=True)


def _class_weights(frame: pd.DataFrame) -> np.ndarray:
    """Use all training rows while giving each outcome class equal total weight."""
    y=frame.target_restatement.to_numpy(dtype=np.int8)
    counts=np.bincount(y,minlength=2)
    if np.any(counts==0):
        raise ValueError("Both outcome classes are required to compute training weights")
    weights=np.where(y==1,len(y)/(2*counts[1]),len(y)/(2*counts[0]))
    return weights.astype(np.float32)


def temporal_holdout_mask(frame: pd.DataFrame, fraction: float = .20) -> tuple[np.ndarray, int]:
    """Hold the latest filing years out of fitting and all model selection."""
    if not 0 < fraction < .5:
        raise ValueError("fraction must be between zero and one half")
    years=np.sort(pd.to_datetime(frame.filed_date,errors="raise").dt.year.unique())
    if len(years)<5:
        raise ValueError("At least five filing years are required for a future-period holdout")
    n_hold=max(1,int(np.ceil(len(years)*fraction)))
    start_year=int(years[-n_hold])
    return pd.to_datetime(frame.filed_date).dt.year.to_numpy()>=start_year,start_year


def _report(y, p, threshold):
    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,average_precision_score
    pred=(np.asarray(p)>=threshold).astype(int)
    y=np.asarray(y).astype(int)
    return {"n":int(len(y)),"positives":int(y.sum()),"negatives":int((y==0).sum()),
            "accuracy":float(accuracy_score(y,pred)),
            "precision":float(precision_score(y,pred,zero_division=0)),
            "recall":float(recall_score(y,pred,zero_division=0)),
            "f1":float(f1_score(y,pred,zero_division=0)),
            "roc_auc":float(roc_auc_score(y,p)) if len(np.unique(y))==2 else None,
            "pr_auc":float(average_precision_score(y,p)) if len(np.unique(y))==2 else None,
            "threshold":float(threshold)}


def _file_sha256(path: Path) -> str:
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""):
            digest.update(block)
    return digest.hexdigest()


def _assert_cuda_booster(model) -> None:
    """Fail if a fitted XGBoost member did not retain its CUDA device setting."""
    config=json.loads(model.get_booster().save_config())
    devices=[]
    def visit(node):
        if isinstance(node,dict):
            for key,value in node.items():
                if key=="device": devices.append(str(value))
                visit(value)
        elif isinstance(node,list):
            for value in node: visit(value)
    visit(config)
    if not any(value.startswith("cuda") for value in devices):
        raise RuntimeError(f"XGBoost did not confirm CUDA after fitting; device settings={devices}")


def _assert_catboost_gpu(model) -> None:
    params=model.get_all_params()
    if str(params.get("task_type","")).upper()!="GPU":
        raise RuntimeError("CatBoost did not confirm task_type=GPU after fitting")


def _best_threshold(y, p):
    candidates=np.unique(np.r_[0.01,np.asarray(p),0.99])
    scored=[]
    for threshold in candidates:
        r=_report(y,p,threshold)
        minimum=min(r[k] for k in ("accuracy","precision","recall","f1"))
        scored.append((minimum,-abs(float(threshold)-.5),float(threshold)))
    return max(scored)[2]


def _bootstrap_metric_intervals(y, p, threshold, seed=20261020, replicates=1000):
    """Stratified percentile intervals for metrics on the stated test sample."""
    y=np.asarray(y).astype(int); p=np.asarray(p,dtype=float)
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg):
        return {}
    rng=np.random.default_rng(seed)
    values={name:[] for name in ("accuracy","precision","recall","f1","roc_auc","pr_auc")}
    for _ in range(replicates):
        idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        report=_report(y[idx],p[idx],threshold)
        for name in values:
            if report[name] is not None:
                values[name].append(report[name])
    return {name:{"lower_95":float(np.quantile(scores,.025)),
                  "upper_95":float(np.quantile(scores,.975))}
            for name,scores in values.items() if scores}


def _paired_auc_difference(y, comparison, baseline, seed=20261020, replicates=1000):
    """Paired stratified bootstrap for the incremental value of a feature view."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    y=np.asarray(y).astype(int)
    comparison=np.asarray(comparison,dtype=float)
    baseline=np.asarray(baseline,dtype=float)
    pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if not len(pos) or not len(neg):
        return {}
    rng=np.random.default_rng(seed)
    point={"roc_auc_difference":float(roc_auc_score(y,comparison)-roc_auc_score(y,baseline)),
           "pr_auc_difference":float(average_precision_score(y,comparison)-average_precision_score(y,baseline))}
    samples={name:[] for name in point}
    for _ in range(replicates):
        idx=np.r_[rng.choice(pos,len(pos),replace=True),rng.choice(neg,len(neg),replace=True)]
        samples["roc_auc_difference"].append(
            float(roc_auc_score(y[idx],comparison[idx])-roc_auc_score(y[idx],baseline[idx])))
        samples["pr_auc_difference"].append(
            float(average_precision_score(y[idx],comparison[idx])-
                  average_precision_score(y[idx],baseline[idx])))
    return {name:{"estimate":point[name],"lower_95":float(np.quantile(values,.025)),
                  "upper_95":float(np.quantile(values,.975))}
            for name,values in samples.items()}


def _save_shap_importance(model, engine, features, labels, feature_names, output_dir):
    """Write local SHAP rankings and feature-family summaries for the holdout."""
    if engine=="xgboost":
        import xgboost as xgb
        contributions=model.get_booster().predict(xgb.DMatrix(features),pred_contribs=True)[:,:-1]
    else:
        from catboost import Pool
        contributions=model.get_feature_importance(Pool(features,label=np.asarray(labels).astype(int)),
                                                   type="ShapValues")[:,:-1]
    if contributions.shape[1] != len(feature_names):
        raise ValueError("SHAP output does not align with the feature-name schema")
    importance=pd.DataFrame({"feature":feature_names,
                             "mean_abs_shap":np.mean(np.abs(contributions),axis=0)})
    importance["feature_group"]=importance.feature.str.split("__",n=1).str[0]
    importance.sort_values("mean_abs_shap",ascending=False).to_csv(
        Path(output_dir)/"feature_importance_shap.csv",index=False)
    grouped=(importance.groupby("feature_group",as_index=False)
             .agg(feature_count=("feature","size"),
                  total_mean_abs_shap=("mean_abs_shap","sum"),
                  mean_abs_shap_per_feature=("mean_abs_shap","mean"))
             .sort_values("total_mean_abs_shap",ascending=False))
    grouped.to_csv(Path(output_dir)/"feature_group_importance_shap.csv",index=False)


def train_gpu(dataset_path: Path, output_dir: Path, seed: int = 20261020,
              archive_path: Path | None = None, target_column: str = "target_restatement",
              target_definition: str | None = None) -> dict:
    """Compare GPU models for a named target with an optional full-archive text view.

    Internally, the established training implementation uses ``target_restatement``
    as a compatibility alias. The requested target column is retained in outputs,
    named in the report, and all other target/provenance fields are excluded from
    predictors.
    """
    import torch
    import xgboost as xgb
    from catboost import CatBoostClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Select an NVIDIA GPU runtime; CPU fallback is disabled.")
    if not xgb.build_info().get("USE_CUDA",False):
        raise RuntimeError("Installed XGBoost was built without CUDA; reinstall the Colab GPU dependencies.")
    output_dir=Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    frame=pd.read_csv(dataset_path,compression="gzip",dtype={"adsh":str,"cik":str},low_memory=False)
    if target_column not in frame:
        raise ValueError(f"Target column {target_column!r} is missing from the prepared data.")
    frame=frame.loc[pd.to_numeric(frame[target_column],errors="coerce").notna()].copy()
    target_values=pd.to_numeric(frame[target_column],errors="raise")
    if not target_values.isin([0,1]).all():
        raise ValueError(f"Target column {target_column!r} must contain only binary values.")
    if target_column != "target_restatement":
        frame["target_restatement"]=target_values.astype("int8")
    else:
        frame["target_restatement"]=target_values.astype("int8")
    target_definition = target_definition or target_column
    frame["cik_norm"]=frame.cik.astype(str).str.lstrip("0").replace("","0")
    frame["filed_year"]=pd.to_datetime(frame.filed_date).dt.year
    frame["_embed_pos"]=np.arange(len(frame),dtype=np.int64)
    classes=frame.target_restatement.value_counts()
    if len(classes)<2 or int(classes.min())<10:
        raise ValueError(f"Need at least 10 rows per class to begin; counts={classes.to_dict()}.")
    groups_per_class=frame.groupby("target_restatement").cik_norm.nunique()
    n_splits=min(5,int(groups_per_class.min()))
    if n_splits<3:
        raise ValueError(f"Need at least 3 distinct issuers per class; issuer counts={groups_per_class.to_dict()}.")
    # Future filing years are excluded from model and threshold selection.
    held_mask,holdout_start_year=temporal_holdout_mask(frame,.20)
    embeds,encoder_meta=_embed_sections(frame)
    archive_meta=None
    if archive_path is not None:
        archive_embeddings,archive_meta,archive_vectorizer,archive_svd=_archive_topic_features(
            frame,archive_path,holdout_start_year)
        embeds.update(archive_embeddings)
        import joblib
        joblib.dump(archive_vectorizer,output_dir/"archive_tfidf_vectorizer.joblib",compress=3)
        joblib.dump(archive_svd,output_dir/"archive_topic_svd.joblib",compress=3)
    dev=frame.loc[~held_mask].reset_index(drop=True)
    hold=frame.loc[held_mask].reset_index(drop=True)
    if dev.target_restatement.nunique()<2:
        raise ValueError("Earlier filing years contain only one outcome class; add reviewed labels across more years.")
    dev_groups=dev.cik_norm.to_numpy()
    dev_groups_per_class=dev.groupby("target_restatement").cik_norm.nunique()
    dev_splits=min(n_splits,int(dev_groups_per_class.min()))
    if dev_splits<3:
        raise ValueError(f"Earlier years need at least 3 distinct issuers per class for grouped CV; counts={dev_groups_per_class.to_dict()}.")
    splitter=StratifiedGroupKFold(n_splits=dev_splits,shuffle=True,random_state=seed)
    # The optional full_archive view adds unsupervised topics learned from the
    # available pre-holdout corpus; labels remain limited to reviewed outcomes.
    views=("financial","mda","interpretable","fusion")
    if archive_path is not None:
        views += ("full_archive",)
    configs=[("xgboost",view,3,240) for view in views]
    configs += [("catboost",view,4,300) for view in views]
    cv_results=[]
    config_failures=[]
    oof={}
    for model_id,(engine,view,depth,trees) in enumerate(configs):
        probs=np.full(len(dev),np.nan,dtype=float)
        failed=False
        thresholds=[]
        for fold,(tr_idx,va_idx) in enumerate(splitter.split(dev,dev.target_restatement,dev_groups)):
            # Retain every development row; class weights balance the loss.
            tr=dev.iloc[tr_idx].reset_index(drop=True)
            va=_balance(dev.iloc[va_idx],seed+fold*20+1)
            if tr.empty or va.empty or va.target_restatement.nunique()<2:
                continue
            # Keep original frame row indexes through accession lookup after balanced sampling.
            train_lookup=dev.set_index("adsh")["_embed_pos"]
            tr_pos=train_lookup.loc[tr.adsh.astype(str)].to_numpy(dtype=int)
            va_pos=train_lookup.loc[va.adsh.astype(str)].to_numpy(dtype=int)
            xtr,xva=_features(tr,va,view,embeds,tr_pos,va_pos)
            train_weights=_class_weights(tr)
            if engine=="xgboost":
                model=xgb.XGBClassifier(n_estimators=trees,max_depth=depth,learning_rate=.035,
                    min_child_weight=4,subsample=.85,colsample_bytree=.85,reg_lambda=8,
                    objective="binary:logistic",eval_metric="logloss",tree_method="hist",
                    device="cuda",random_state=seed+fold,n_jobs=2)
                model.fit(xtr,tr.target_restatement.astype(int),sample_weight=train_weights)
                _assert_cuda_booster(model)
            else:
                try:
                    model=CatBoostClassifier(iterations=trees,depth=depth,learning_rate=.035,
                        loss_function="Logloss",eval_metric="Logloss",task_type="GPU",devices="0",
                        l2_leaf_reg=8,random_seed=seed+fold,verbose=False,allow_writing_files=False)
                    model.fit(xtr,tr.target_restatement.astype(int),sample_weight=train_weights,
                              verbose=False)
                    _assert_catboost_gpu(model)
                except Exception as exc:
                    config_failures.append({"engine":engine,"view":view,"depth":depth,
                                            "trees":trees,"fold":fold,
                                            "error":f"{type(exc).__name__}: {exc}"[:500]})
                    failed=True
                    break
            p=model.predict_proba(xva)[:,1]
            # Map probabilities to the sampled validation subset; use distinct
            # IDs so no issuer contributes train/test rows in this fold.
            for adsh,prob in zip(va.adsh.astype(str),p):
                loc=dev.index[dev.adsh.astype(str).eq(adsh)]
                if len(loc): probs[loc[0]]=prob
        if failed:
            continue
        usable=np.isfinite(probs)
        if usable.sum()<20 or dev.loc[usable,"target_restatement"].nunique()<2:
            continue
        balanced_oof=_balance(dev.loc[usable].assign(_p=probs[usable]),seed+model_id)
        ids=balanced_oof.adsh.astype(str)
        # Reconstruct aligned predictions from the OOF accession map.
        p_lookup=dict(zip(dev.loc[usable].adsh.astype(str),probs[usable]))
        balanced_prob=np.asarray([p_lookup[a] for a in ids])
        threshold=_best_threshold(balanced_oof.target_restatement.to_numpy(),balanced_prob)
        report=_report(balanced_oof.target_restatement.to_numpy(),balanced_prob,threshold)
        report.update({"engine":engine,"view":view,"max_depth":depth,"n_estimators":trees,
                       "oof_rows":int(usable.sum()),"selection_metric_min":min(report[k] for k in ("accuracy","precision","recall","f1"))})
        cv_results.append(report)
        oof[(engine,view,depth,trees)]=(probs,threshold)
    if not cv_results:
        raise ValueError("No valid issuer-grouped CV folds. Add reviewed outcomes from more issuers/years.")
    # Select on issuer-grouped development PR-AUC; the forward holdout is untouched.
    leaderboard=pd.DataFrame(cv_results).sort_values(["pr_auc","roc_auc"],ascending=False)
    best=leaderboard.iloc[0]
    best_key=(str(best.engine),str(best.view),int(best.max_depth),int(best.n_estimators))
    threshold=float(oof[best_key][1])
    best_by_config={}
    for engine in ("xgboost","catboost"):
        for view in views:
            candidates=leaderboard.loc[leaderboard.view.eq(view)&leaderboard.engine.eq(engine)]
            if candidates.empty:
                continue
            view_best=candidates.iloc[0]
            key=(str(view_best.engine),str(view_best.view),int(view_best.max_depth),
                 int(view_best.n_estimators))
            best_by_config[f"{engine}/{view}"]={"row":view_best,"key":key,
                                                 "threshold":float(oof[key][1])}
    hold_bal=_balance(hold,seed+999)
    result={"gpu":torch.cuda.get_device_name(0),"xgboost":xgb.__version__,"rows":len(frame),
            "target_column":target_column,"target_definition":target_definition,
            "horizon_years":(int(match.group(1)) if (match:=re.search(r"h([135])y",Path(dataset_path).name)) else None),
            "positives":int(frame.target_restatement.sum()),"negatives":int((frame.target_restatement==0).sum()),
            "current_mda_coverage":float(frame.mda_status.isin(["ok","ok_truncated"]).mean()),
            "prior_mda_coverage":float(frame.prior_mda_status.isin(["ok","ok_truncated"]).mean()),
            "issuer_groups":int(frame.cik_norm.nunique()),"holdout_rows":len(hold),
            "future_holdout_start_year":holdout_start_year,
            "dataset_sha256":_file_sha256(Path(dataset_path)),
            "study_code_sha256":_file_sha256(Path(__file__)),
            "embedding_encoder":encoder_meta,
            "financial_trajectory_features":{
                "feature_count":int(sum(str(c).startswith("trajectory__") for c in frame.columns)),
                "definition":"Current and up to three strictly earlier same-issuer/same-form filings; ratio slopes, dispersion, range, and latest step.",
                "future_or_same_day_history_used":False},
            "full_archive_text_features":archive_meta,
            "library_versions":{name:importlib.metadata.version(dist) for name,dist in
                (("torch","torch"),("sentence_transformers","sentence-transformers"),
                 ("scikit_learn","scikit-learn"),("xgboost","xgboost-cu12"),
                 ("catboost","catboost"))},
            "holdout_positive_issuers":int(hold.loc[hold.target_restatement.eq(1),"cik_norm"].nunique()),
            "holdout_negative_issuers":int(hold.loc[hold.target_restatement.eq(0),"cik_norm"].nunique()),
            "selected_model":{"engine":best_key[0],"view":best_key[1],"max_depth":best_key[2],"n_estimators":best_key[3],"threshold":threshold},
            "selected_models_by_view":{name:{"engine":item["key"][0],"view":item["key"][1],
                "max_depth":item["key"][2],"n_estimators":item["key"][3],
                "threshold":item["threshold"]} for name,item in best_by_config.items()},
            "development_cv_1_to_1":{k:v for k,v in best.to_dict().items()},
            "model_selection_metric":"issuer_grouped_development_pr_auc",
            "gpu_model_failures":config_failures,
            "holdout_results_by_view":{},
            "balanced_holdout_1_to_1":None,"curated_holdout_all_rows":None,
            "holdout_adequate_for_claim":bool(len(hold_bal)>=60 and hold_bal.target_restatement.sum()>=30 and
                    hold_bal.loc[hold_bal.target_restatement.eq(1),"cik_norm"].nunique()>=10 and
                    hold_bal.loc[hold_bal.target_restatement.eq(0),"cik_norm"].nunique()>=10),
            "metric_goal_all_four_ge_85":False,
            "note":"Development CV selects the feature set and threshold. The latest filing-year holdout is evaluated once. Development folds are issuer-grouped. The reviewed queue is a curated case-control sample, not natural prevalence."
                +(f" The archive proxy labels were ignored; TF-IDF/SVD was fit only on filings before {holdout_start_year}."
                  if archive_meta else "")}
    # Evaluate every prespecified feature view on the same untouched future years.
    # Every view's model and threshold come from grouped development data only.
    if not hold.empty and hold.target_restatement.nunique()==2 and hold_bal.target_restatement.nunique()==2:
        train=dev.copy()
        lookup=dev.set_index("adsh")["_embed_pos"]
        train_pos=lookup.loc[train.adsh.astype(str)].to_numpy(dtype=int)
        hold_pos=frame.index[held_mask]
        train_weights=_class_weights(train)
        holdout_probabilities={}
        prediction_frames=[]
        output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
        for view_index,(model_name,selected) in enumerate(best_by_config.items()):
            engine,_,depth,trees=selected["key"]
            view=selected["key"][1]
            view_threshold=selected["threshold"]
            xtr,xh,feature_names=_features(train,hold,view,embeds,train_pos,hold_pos,
                                           return_names=True)
            if engine=="xgboost":
                final=xgb.XGBClassifier(n_estimators=trees,max_depth=depth,learning_rate=.035,
                    min_child_weight=4,subsample=.85,colsample_bytree=.85,reg_lambda=8,
                    objective="binary:logistic",eval_metric="logloss",tree_method="hist",device="cuda",
                    random_state=seed+view_index,n_jobs=2)
                final.fit(xtr,train.target_restatement.astype(int),sample_weight=train_weights)
                _assert_cuda_booster(final)
                model_path=output_dir/f"model_{view}_{engine}.json"
            else:
                final=CatBoostClassifier(iterations=trees,depth=depth,learning_rate=.035,
                    loss_function="Logloss",eval_metric="Logloss",task_type="GPU",devices="0",
                    l2_leaf_reg=8,random_seed=seed+view_index,verbose=False,allow_writing_files=False)
                final.fit(xtr,train.target_restatement.astype(int),sample_weight=train_weights,
                          verbose=False)
                _assert_catboost_gpu(final)
                model_path=output_dir/f"model_{view}_{engine}.cbm"
            hold_probs=final.predict_proba(xh)[:,1]
            holdout_probabilities[model_name]=hold_probs
            prob_lookup=dict(zip(hold.adsh.astype(str),hold_probs))
            balanced_prob=np.asarray([prob_lookup[a] for a in hold_bal.adsh.astype(str)])
            balanced_report=_report(hold_bal.target_restatement,balanced_prob,view_threshold)
            all_report=_report(hold.target_restatement,hold_probs,view_threshold)
            result["holdout_results_by_view"][model_name]={
                "engine":engine,"view":view,"threshold":view_threshold,
                "balanced_holdout_1_to_1":balanced_report,
                "balanced_holdout_1_to_1_ci95":_bootstrap_metric_intervals(
                    hold_bal.target_restatement,balanced_prob,view_threshold,seed+3000+view_index),
                "curated_holdout_all_rows":all_report,
                "curated_holdout_all_rows_ci95":_bootstrap_metric_intervals(
                    hold.target_restatement,hold_probs,view_threshold,seed+4000+view_index)}
            prediction_frames.append(pd.DataFrame({
                "adsh":hold.adsh.astype(str),target_column:hold.target_restatement.astype(int),
                "model_view":view,"model_engine":engine,"threshold":view_threshold,
                "probability":hold_probs,"predicted":(hold_probs>=view_threshold).astype(int),
                "cik_group_hash":[hashlib.sha256(str(v).encode()).hexdigest()[:16]
                                  for v in hold.cik_norm]}))
            final.save_model(model_path)
            if selected["key"]==best_key:
                result["balanced_holdout_1_to_1"]=balanced_report
                result["balanced_holdout_1_to_1_ci95"]=result["holdout_results_by_view"][model_name]["balanced_holdout_1_to_1_ci95"]
                result["curated_holdout_all_rows"]=all_report
                result["curated_holdout_all_rows_ci95"]=result["holdout_results_by_view"][model_name]["curated_holdout_all_rows_ci95"]
                result["metric_goal_all_four_ge_85"]=all(balanced_report[k]>=.85 for k in
                    ("accuracy","precision","recall","f1"))
                sensitivity=[]
                for draw in range(50):
                    sample=_balance(hold,seed+2000+draw)
                    if sample.empty or sample.target_restatement.nunique()<2:
                        continue
                    sampled_prob=np.asarray([prob_lookup[a] for a in sample.adsh.astype(str)])
                    sensitivity.append(_report(sample.target_restatement,sampled_prob,view_threshold))
                report_metrics=("accuracy","precision","recall","f1","roc_auc","pr_auc")
                result["matched_sample_sensitivity_50"]={
                    "draws":len(sensitivity),
                    "metrics":{name:{"median":float(np.nanmedian([row[name] for row in sensitivity if row[name] is not None])),
                                      "p05":float(np.nanquantile([row[name] for row in sensitivity if row[name] is not None],.05)),
                                      "p95":float(np.nanquantile([row[name] for row in sensitivity if row[name] is not None],.95))}
                               for name in report_metrics if any(row[name] is not None for row in sensitivity)},
                    "fraction_all_four_ge_85":float(np.mean([all(row[k]>=.85 for k in ("accuracy","precision","recall","f1"))
                                                              for row in sensitivity])) if sensitivity else None,
                    "note":"Repeated 1:1 matching sensitivity, not a confidence interval."}
                try:
                    _save_shap_importance(final,engine,xh,hold.target_restatement,
                                          feature_names,output_dir)
                    result["shap_explanations"]="feature_importance_shap.csv and feature_group_importance_shap.csv"
                except Exception as exc:
                    result["shap_explanations_error"]=f"{type(exc).__name__}: {exc}"[:500]
        result["paired_holdout_auc_differences"]={}
        for engine in ("xgboost","catboost"):
            if f"{engine}/financial" not in holdout_probabilities:
                continue
            baseline=holdout_probabilities[f"{engine}/financial"]
            baseline_balanced=np.asarray([dict(zip(hold.adsh.astype(str),baseline))[a]
                                          for a in hold_bal.adsh.astype(str)])
            for view_index,view in enumerate(views):
                if view=="financial":
                    continue
                model_name=f"{engine}/{view}"
                if model_name not in holdout_probabilities:
                    continue
                view_prob=holdout_probabilities[model_name]
                view_balanced=np.asarray([dict(zip(hold.adsh.astype(str),view_prob))[a]
                                          for a in hold_bal.adsh.astype(str)])
                result["paired_holdout_auc_differences"][model_name]={
                    "balanced_holdout_1_to_1":_paired_auc_difference(
                        hold_bal.target_restatement,view_balanced,baseline_balanced,
                        seed+5000+view_index),
                    "curated_holdout_all_rows":_paired_auc_difference(
                        hold.target_restatement,view_prob,baseline,seed+6000+view_index)}
        pd.concat(prediction_frames,ignore_index=True).to_csv(
            output_dir/"future_holdout_predictions.csv",index=False)
    output_dir=Path(output_dir); output_dir.mkdir(parents=True,exist_ok=True)
    leaderboard.to_csv(output_dir/"grouped_cv_leaderboard.csv",index=False)
    (output_dir/"metrics.json").write_text(json.dumps(result,indent=2,default=str),encoding="utf-8")
    def pct(value):
        return "NA" if value is None else f"{100*float(value):.1f}%"
    balanced=result.get("balanced_holdout_1_to_1") or {}
    all_rows=result.get("curated_holdout_all_rows") or {}
    holdout_views=result.get("holdout_results_by_view",{})
    balanced_rows=[]
    all_rows_lines=[]
    for view,view_result in holdout_views.items():
        for metric_key,destination in (("balanced_holdout_1_to_1",balanced_rows),
                                       ("curated_holdout_all_rows",all_rows_lines)):
            metrics=view_result.get(metric_key,{})
            destination.append(
                f"| {view} | {metrics.get('n',0)} | {pct(metrics.get('accuracy'))} | "
                f"{pct(metrics.get('precision'))} | {pct(metrics.get('recall'))} | "
                f"{pct(metrics.get('f1'))} | {pct(metrics.get('roc_auc'))} | "
                f"{pct(metrics.get('pr_auc'))} |")
    paired_lines=[]
    for view,comparison in result.get("paired_holdout_auc_differences",{}).items():
        values=comparison.get("balanced_holdout_1_to_1",{})
        pr=values.get("pr_auc_difference",{})
        roc=values.get("roc_auc_difference",{})
        paired_lines.append(f"- `{view}` minus financial baseline, matched holdout: "
            f"PR-AUC difference {pct(pr.get('estimate'))} "
            f"(95% CI {pct(pr.get('lower_95'))} to {pct(pr.get('upper_95'))}); "
            f"ROC-AUC difference {pct(roc.get('estimate'))} "
            f"(95% CI {pct(roc.get('lower_95'))} to {pct(roc.get('upper_95'))}).")
    report=(f"# MD&A outcome study: {target_column}\n\n"
            f"**Outcome definition:** {target_definition}\n\n"
            "The evaluation uses issuer-grouped development folds and a later-year holdout. "
            "All reported metrics refer to the named outcome above.\n\n"
            f"- Reviewed model rows: {result.get('rows', 0)} "
            f"({result.get('positives', 0)} positive; {result.get('negatives', 0)} negative).\n"
            f"- Forward holdout begins in {result.get('future_holdout_start_year', 'NA')}; "
            f"{result.get('holdout_rows', 0)} holdout rows before 1:1 matching.\n"
            + (f"- Full archive text features: {result['full_archive_text_features'].get('archive_rows', 0)} documents; "
               f"{result['full_archive_text_features'].get('archive_fit_rows', 0)} pre-holdout documents fit the text transform "
               f"through {result['full_archive_text_features'].get('fit_through_year', 'NA')}; legacy proxy labels were ignored.\n"
               if result.get('full_archive_text_features') else "")
            + f"- Selected model: {result.get('selected_model', {}).get('engine', 'NA')} / "
            f"{result.get('selected_model', {}).get('view', 'NA')}; selected using issuer-grouped development PR-AUC.\n\n"
            "## Forward holdout: year/form-matched 1:1\n\n"
            "| Feature view | n | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(balanced_rows) + "\n\n"
            "## Forward holdout: all curated rows\n\n"
            "| Feature view | n | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |\n"
            "|---|---:|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(all_rows_lines) + "\n\n"
            "## Paired feature-view differences\n\n"
            + ("\n".join(paired_lines) if paired_lines else "No paired comparison was produced.") + "\n\n"
            "Bootstrap 95% intervals are in `metrics.json`. Feature attributions are in the SHAP CSV files.\n\n"
            "## Limits\n\n"
            "The source cohort is enriched and sampled. The 1:1 results apply only to the matched case-control "
            "sample, and precision does not estimate population precision. If the target uses AI-assisted "
            "outcome screening, report that labeling method and its uncertainty.\n")
    (output_dir/"study_report.md").write_text(report,encoding="utf-8")
    return result

