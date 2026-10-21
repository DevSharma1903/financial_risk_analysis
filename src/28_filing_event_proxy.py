"""Build an auditable filing-event proxy cohort from the review queue.

The primary proxy endpoint is an observed later same-period 10-K/A or 10-Q/A
candidate within three years. A separate composite sensitivity endpoint also
counts reviewed issuer-level restatement/revision events. These are not labels
for confirmed restatement prediction. Source sampling and unresolved outcomes
are recorded and excluded from the relevant model target.
"""
from __future__ import annotations

from contextlib import closing
from pathlib import Path
import gzip
import hashlib
import sqlite3

import numpy as np
import pandas as pd


HORIZONS = (1, 3, 5)
CANDIDATE_GROUP = "same_period_amendment_candidate"
CONTROL_GROUP = "no_same_period_amendment_control"


def derive_targets(ledger: pd.DataFrame, horizon_years: int = 3) -> pd.DataFrame:
    """Derive two transparent targets and keep unresolved composite rows null."""
    if horizon_years not in HORIZONS:
        raise ValueError(f"horizon_years must be one of {HORIZONS}")
    required = {
        "original_adsh", "original_filed_date", "sampling_group",
        "candidate_event_adsh", "candidate_event_date", "review_status",
        "event_type", "event_date", "outcome_followup_through",
        "review_queue_horizon_years",
    }
    missing = required - set(ledger.columns)
    if missing:
        raise ValueError(f"Review queue is missing required columns: {sorted(missing)}")
    out = ledger.copy()
    out["original_adsh"] = out.original_adsh.fillna("").astype(str).str.strip()
    if out.original_adsh.eq("").any() or out.original_adsh.duplicated().any():
        raise ValueError("Each review queue row needs one unique original_adsh")
    if not pd.to_numeric(out.review_queue_horizon_years, errors="coerce").eq(horizon_years).all():
        raise ValueError("Review queue horizon does not match the selected label horizon")
    groups = out.sampling_group.fillna("").astype(str)
    if not groups.isin([CANDIDATE_GROUP, CONTROL_GROUP]).all():
        raise ValueError("Unexpected sampling_group value in the review queue")
    original = pd.to_datetime(out.original_filed_date, errors="coerce")
    if original.isna().any():
        raise ValueError("Every review queue row needs a parseable original filing date")
    horizon_end = original + pd.DateOffset(years=horizon_years)
    candidate_id = out.candidate_event_adsh.fillna("").astype(str).str.strip()
    candidate_date = pd.to_datetime(out.candidate_event_date, errors="coerce")
    candidates = groups.eq(CANDIDATE_GROUP)
    controls = groups.eq(CONTROL_GROUP)
    if (candidates & candidate_id.eq("")).any() or (controls & candidate_id.ne("")).any():
        raise ValueError("Candidate/control group disagrees with candidate accession evidence")
    if (candidates & candidate_date.isna()).any():
        raise ValueError("Every amendment candidate needs its event date")
    if (candidates & candidate_date.lt(original)).any():
        raise ValueError("A candidate amendment cannot precede its original filing")
    if (candidates & candidate_date.gt(horizon_end)).any():
        raise ValueError("A candidate amendment falls outside the selected outcome horizon")

    # Target A: the occurrence of the later same-period amended filing itself.
    out["target_same_period_amendment_proxy"] = candidates.astype("int8")
    out["label_provenance"] = np.where(candidates, "observed_same_period_amendment_candidate", "control_without_candidate")

    # Target B: broader observed filing event. Candidate amendments are positive
    # even when the restatement screen is pending. Controls require a reviewed,
    # mature outcome; pending controls remain unlabeled.
    review_status = out.review_status.fillna("").astype(str).str.strip().str.lower()
    event_type = out.event_type.fillna("").astype(str).str.strip().str.lower()
    event_date = pd.to_datetime(out.event_date, errors="coerce")
    followup = pd.to_datetime(out.outcome_followup_through, errors="coerce")
    composite = pd.Series(pd.NA, index=out.index, dtype="Int8")
    composite.loc[candidates] = 1
    provenance = pd.Series("unresolved_control_outcome", index=out.index, dtype="object")
    provenance.loc[candidates] = "observed_same_period_amendment_candidate"
    reviewed_control = controls & review_status.eq("reviewed")
    control_restatement = reviewed_control & event_type.eq("restatement")
    restatement_inside = control_restatement & event_date.notna() & event_date.le(horizon_end)
    restatement_after = control_restatement & event_date.notna() & event_date.gt(horizon_end) & followup.ge(event_date)
    reviewed_no_event = (reviewed_control & event_type.eq("no_confirmed_restatement") &
                         followup.notna() & followup.ge(horizon_end))
    other_reviewed_event = (reviewed_control & event_type.isin(["revision_not_restatement", "other_amendment"]) &
                            event_date.notna() & event_date.le(horizon_end))
    other_reviewed_after = (reviewed_control & event_type.isin(["revision_not_restatement", "other_amendment"]) &
                            event_date.notna() & event_date.gt(horizon_end) & followup.ge(event_date))
    composite.loc[restatement_inside | other_reviewed_event] = 1
    composite.loc[restatement_after | reviewed_no_event | other_reviewed_after] = 0
    provenance.loc[restatement_inside] = "reviewed_restatement_within_horizon"
    provenance.loc[other_reviewed_event] = "reviewed_amendment_or_revision_within_horizon"
    provenance.loc[restatement_after] = "reviewed_restatement_after_horizon"
    provenance.loc[reviewed_no_event] = "reviewed_no_confirmed_restatement_through_horizon"
    provenance.loc[other_reviewed_after] = "reviewed_amendment_or_revision_after_horizon"
    out["target_amendment_or_screened_event"] = composite
    out["label_provenance"] = provenance
    out["horizon_end"] = horizon_end.dt.strftime("%Y-%m-%d")
    return out


def validate_maturity(proxy: pd.DataFrame, cohort: pd.DataFrame, horizon_years: int) -> None:
    """Ensure the prepared source's outcome cutoff matures all sampled filings."""
    if "label_observation_cutoff" not in proxy:
        raise ValueError("Prepared feature table needs label_observation_cutoff")
    cutoffs = pd.to_datetime(proxy.label_observation_cutoff, errors="coerce").dropna().unique()
    if len(cutoffs) != 1:
        raise ValueError("Expected one common label_observation_cutoff")
    cutoff = pd.Timestamp(cutoffs[0])
    filed = pd.to_datetime(cohort.original_filed_date, errors="raise")
    if (filed + pd.DateOffset(years=horizon_years) > cutoff).any():
        raise ValueError("Some sampled filings lack the full selected outcome horizon")


def collect_mda(proxy: pd.DataFrame, cohort: pd.DataFrame, db_path: Path,
                user_agent: str, max_new: int = 500, client=None,
                study=None, evidence=None, advanced=None) -> dict:
    """Fetch original and prior comparable MD&A, resuming an existing SQLite cache."""
    if not user_agent or "@" not in user_agent:
        raise ValueError("Supply a real identifying SEC User-Agent with contact email")
    if study is None or evidence is None or advanced is None:
        raise ValueError("Pass the loaded study, SEC evidence, and MD&A extractor modules")
    study.validate_proxy_features(proxy)
    proxy = proxy.copy()
    proxy["filed_date"] = pd.to_datetime(proxy.filed_date, errors="raise")
    proxy["cik"] = proxy.cik.astype(str).str.lstrip("0").replace("", "0")
    targets = cohort[["original_adsh", "original_filed_date"]].copy()
    targets = targets.merge(proxy[["adsh", "cik", "form", "filed_date"]].rename(columns={
        "adsh":"original_adsh", "form":"original_form", "filed_date":"proxy_filed_date"}),
        on="original_adsh", how="left", validate="one_to_one")
    if targets.cik.isna().any():
        raise ValueError("Some review queue originals are absent from the prepared SEC feature table")
    if not targets.form.isin(["10-K", "10-Q"]).all():
        raise ValueError("Only original 10-K/10-Q filings may supply predictors")
    if not pd.to_datetime(targets.original_filed_date, errors="raise").eq(targets.proxy_filed_date).all():
        raise ValueError("Review queue original filing date disagrees with its accession record")
    link_targets = targets[["original_adsh", "cik", "original_form", "proxy_filed_date"]].rename(
        columns={"proxy_filed_date":"original_filed_date"})
    links = study._prior_filing_links(proxy, link_targets)
    target_manifest = targets.rename(columns={"original_adsh":"adsh", "original_form":"form",
        "proxy_filed_date":"filed_date"})[["adsh", "cik", "form", "filed_date"]]
    prior_ids = set(links.prior_adsh) - {""}
    prior_manifest = proxy.loc[proxy.adsh.astype(str).isin(prior_ids),
        ["adsh", "cik", "form", "filed_date"]].copy()
    manifest = pd.concat([target_manifest, prior_manifest], ignore_index=True).drop_duplicates("adsh")
    manifest["adsh"] = manifest.adsh.astype(str)
    manifest["cik"] = manifest.cik.astype(str).str.lstrip("0").replace("", "0")
    manifest["filed_date"] = pd.to_datetime(manifest.filed_date, errors="raise")
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    client = client or evidence.SecClient(user_agent)
    db = study.section_db(db_path)
    done = {row[0] for row in db.execute("SELECT adsh FROM restatement_mda")}
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
        db.commit()
    finally:
        db.close()
    with closing(study.section_db(db_path)) as db:
        states = dict(db.execute("SELECT adsh,status FROM restatement_mda"))
    counts = pd.Series([states.get(a, "pending") for a in manifest.adsh]).value_counts().to_dict()
    return {"target_originals":len(targets), "prior_filings":len(prior_ids),
            "unique_filings":len(manifest), "attempted_this_session":attempted,
            "status_counts":counts, "pending":int(counts.get("pending", 0))}


def prepare_model_data(proxy: pd.DataFrame, cohort: pd.DataFrame, sections_db_path: Path,
                       output_path: Path) -> dict:
    """Join outcomes, point-in-time predictors, and cached current/prior MD&A."""
    proxy = proxy.copy()
    study_features = {"target_same_period_amendment_proxy", "target_amendment_or_screened_event"}
    ids = set(cohort.original_adsh.astype(str))
    part = proxy.loc[proxy.adsh.astype(str).isin(ids)].copy()
    if len(part) != len(cohort):
        raise ValueError("Some labeled originals are missing from the feature table")
    outcome_cols = ["original_adsh", "target_same_period_amendment_proxy",
                    "target_amendment_or_screened_event", "label_provenance", "horizon_end"]
    part = part.merge(cohort[outcome_cols].rename(columns={"original_adsh":"adsh"}),
                      on="adsh", validate="one_to_one")
    part["original_adsh"] = part.adsh.astype(str)
    part["filed_date"] = pd.to_datetime(part.filed_date, errors="raise")
    link_proxy = proxy[["adsh", "cik", "form", "filed_date"]].copy()
    link_proxy["cik"] = link_proxy.cik.astype(str).str.lstrip("0").replace("", "0")
    link_proxy["filed_date"] = pd.to_datetime(link_proxy.filed_date, errors="raise")
    targets = part[["adsh", "cik", "form", "filed_date"]].rename(columns={
        "adsh":"original_adsh", "form":"original_form", "filed_date":"original_filed_date"})
    targets["cik"] = targets.cik.astype(str).str.lstrip("0").replace("", "0")
    # Use a local grouping implementation to avoid pulling outcome labels into links.
    hist = link_proxy.copy()
    hist["cik_norm"] = hist.cik.astype(str).str.lstrip("0").replace("", "0")
    hist["filed_date"] = pd.to_datetime(hist.filed_date, errors="raise")
    hist = hist.sort_values(["cik_norm", "form", "filed_date", "adsh"])
    buckets = {(str(cik), str(form)): group for (cik, form), group in
               hist.groupby(["cik_norm", "form"], sort=False)}
    prior = []
    for row in targets.itertuples(index=False):
        cik = str(row.cik).lstrip("0") or "0"
        group = buckets.get((cik, str(row.original_form)))
        earlier = group.loc[group.filed_date.lt(pd.Timestamp(row.original_filed_date))] if group is not None else None
        prior.append("" if earlier is None or earlier.empty else str(earlier.iloc[-1].adsh))
    part["prior_adsh"] = prior
    with closing(sqlite3.connect(Path(sections_db_path))) as db:
        texts = {adsh:(gzip.decompress(body).decode("utf-8", errors="replace"), status)
                 for adsh, body, status in db.execute("SELECT adsh,body,status FROM restatement_mda")}
    part["mda_text"] = part.adsh.astype(str).map(lambda a:texts.get(a,("","missing"))[0])
    part["mda_status"] = part.adsh.astype(str).map(lambda a:texts.get(a,("","pending"))[1])
    part["prior_mda_text"] = part.prior_adsh.astype(str).map(lambda a:texts.get(a,("","missing"))[0])
    part["prior_mda_status"] = part.prior_adsh.astype(str).map(lambda a:texts.get(a,("","pending"))[1])
    keep_meta = {"adsh", "original_adsh", "cik", "form", "fp", "sic2", "filed_date", "period_date",
        "filed_year", *study_features, "label_provenance", "horizon_end", "prior_adsh",
        "mda_text", "mda_status", "prior_mda_text", "prior_mda_status"}
    excluded_point_in_time_meta = {"adsh", "cik", "period", "filed", "filed_date",
        "filed_year", "period_date", "sic", "afs", "wksi", "feature_schema_version",
        "label_observation_cutoff", "y_amendment_proxy", "days_to_amendment"}
    feature_cols = [c for c in part if c not in keep_meta and c not in excluded_point_in_time_meta and
        not c.startswith(("target_", "label_", "event_", "evidence_", "candidate_",
                          "review_", "sampling_", "outcome_")) and
        c not in {"y_amendment_proxy", "days_to_amendment"}]
    # Explicitly remove legacy labels/outcomes before any model consumes the file.
    forbidden = {"y_amendment_proxy", "days_to_amendment", "candidate_event_adsh",
                 "candidate_event_date", "candidate_event_url", "sampling_group"}
    feature_cols = [c for c in feature_cols if c not in forbidden]
    meta_order = [c for c in ["adsh", "cik", "form", "fp", "sic2", "filed_date",
        "period_date", *study_features, "label_provenance", "horizon_end", "prior_adsh",
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
        "same_period_amendment_positives":int(safe.target_same_period_amendment_proxy.sum()),
        "composite_positives":int(safe.target_amendment_or_screened_event.eq(1).sum()),
        "composite_negatives":int(safe.target_amendment_or_screened_event.eq(0).sum()),
        "composite_unresolved":int(safe.target_amendment_or_screened_event.isna().sum()),
        "current_mda_usable":int(safe.mda_status.isin(["ok", "ok_truncated"]).sum()),
        "prior_mda_usable":int(safe.prior_mda_status.isin(["ok", "ok_truncated"]).sum()),
        "feature_count":len(feature_cols), "excluded_legacy_proxy_columns":sorted(forbidden)}


def summarize(cohort: pd.DataFrame) -> dict:
    """Counts and provenance summary for pre-run inspection and audit."""
    return {
        "rows":int(len(cohort)),
        "sampling_groups":cohort.sampling_group.value_counts().to_dict(),
        "same_period_amendment_proxy":cohort.target_same_period_amendment_proxy.value_counts().to_dict(),
        "amendment_or_screened_event":cohort.target_amendment_or_screened_event.value_counts(dropna=False).to_dict(),
        "provenance":cohort.label_provenance.value_counts().to_dict(),
        "outcome_horizon_years":int(pd.to_numeric(cohort.review_queue_horizon_years, errors="raise").iloc[0]),
    }
