"""Validate the human-reviewed confirmed-restatement ledger.

Pending rows are permitted in the working queue but are never model labels.
This validator checks fields and source provenance, not whether a reviewer's
accounting judgment is correct.
"""
from pathlib import Path
import argparse
import pandas as pd


REQUIRED = {
    "original_adsh", "original_filed_date", "label", "review_status", "event_type",
    "event_date", "outcome_followup_through", "evidence_url", "evidence_excerpt",
    "reviewer", "review_notes",
}
ALLOWED_EVENT_TYPES = {
    "restatement", "revision_not_restatement", "other_amendment",
    "no_confirmed_restatement",
}


def validate_ledger(frame: pd.DataFrame, min_followup_years: int = 0) -> list[str]:
    """Return structural errors; pending rows are allowed but excluded from fitting."""
    errors = []
    if min_followup_years < 0:
        return ["min_followup_years cannot be negative"]
    missing = REQUIRED - set(frame.columns)
    if missing:
        return [f"Missing required columns: {sorted(missing)}"]
    ids = frame.original_adsh.fillna("").astype(str).str.strip()
    if ids.eq("").any():
        errors.append("original_adsh must be present on every queue row")
    if ids[ids.ne("")].duplicated().any():
        errors.append("original_adsh must be unique (one final adjudication per filing)")

    status = frame.review_status.fillna("").astype(str).str.strip().str.lower()
    if (~status.isin(["pending", "reviewed"])).any():
        errors.append("review_status must be pending or reviewed")
    reviewed = frame.loc[status.eq("reviewed")].copy()
    if reviewed.empty:
        return errors

    labels = pd.to_numeric(reviewed.label, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        errors.append("Every reviewed row needs label 0 or 1; pending/unknown rows stay pending")
    event_types = reviewed.event_type.fillna("").astype(str).str.strip()
    if not event_types.isin(ALLOWED_EVENT_TYPES).all():
        errors.append(f"Reviewed event_type must be one of {sorted(ALLOWED_EVENT_TYPES)}")
    expected_label = event_types.eq("restatement").astype(int)
    if labels.notna().all() and not labels.astype(int).eq(expected_label).all():
        errors.append("label must be 1 only for event_type=restatement and 0 otherwise")

    original_dates = pd.to_datetime(reviewed.original_filed_date, errors="coerce")
    event_dates = pd.to_datetime(reviewed.event_date, errors="coerce")
    followup = pd.to_datetime(reviewed.outcome_followup_through, errors="coerce")
    if original_dates.isna().any():
        errors.append("Reviewed rows need a parseable original_filed_date")
    if followup.isna().any():
        errors.append("Every reviewed row needs outcome_followup_through")
    positives = labels.eq(1)
    event_required = event_types.ne("no_confirmed_restatement")
    if (event_required & event_dates.isna()).any():
        errors.append("Reviewed amendment/restatement events need event_date")
    if (positives & event_dates.isna()).any():
        errors.append("Positive restatement rows need event_date")
    if (positives & event_dates.lt(original_dates)).any():
        errors.append("A restatement event cannot precede its original filing")
    if (event_dates.notna() & event_dates.gt(followup)).any():
        errors.append("outcome_followup_through cannot precede the documented event_date")
    if (followup.notna() & original_dates.notna() & followup.lt(original_dates)).any():
        errors.append("outcome_followup_through cannot precede the original filing")

    for column in ("evidence_url", "evidence_excerpt", "reviewer"):
        values = reviewed[column].fillna("").astype(str).str.strip()
        if values.eq("").any():
            errors.append(f"Every reviewed row needs {column}")
    urls = reviewed.evidence_url.fillna("").astype(str).str.lower()
    if (~urls.str.startswith("https://www.sec.gov/")).any():
        errors.append("Evidence URLs must point to a primary SEC.gov filing source")

    if min_followup_years:
        required_through = original_dates + pd.DateOffset(years=min_followup_years)
        # If a confirmed event occurred after the requested horizon, its event
        # date establishes a negative at that horizon, provided it is observed.
        event_after_horizon = event_dates.notna() & event_dates.gt(required_through)
        event_within_horizon = event_types.eq("restatement") & event_dates.le(required_through)
        too_early = followup.lt(required_through) & ~event_after_horizon & ~event_within_horizon
        if too_early.any():
            errors.append(f"Reviewed negatives/early events need at least {min_followup_years} years of outcome follow-up")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="Human-reviewed outcome ledger CSV")
    parser.add_argument("--horizon-years", type=int, default=0,
                        help="Optionally require mature follow-up for this horizon (1, 3, or 5)")
    args = parser.parse_args()
    if not args.ledger.is_file():
        raise FileNotFoundError(args.ledger)
    frame = pd.read_csv(args.ledger, dtype={"original_adsh": str})
    errors = validate_ledger(frame, args.horizon_years)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    status = frame.review_status.fillna("").astype(str).str.lower()
    reviewed = frame.loc[status.eq("reviewed")]
    labels = pd.to_numeric(reviewed.label, errors="coerce").dropna().astype(int)
    print(f"Valid queue: {len(frame):,} rows; reviewed={len(reviewed):,}; pending={(~status.eq('reviewed')).sum():,}.")
    if len(labels):
        print(f"Reviewed outcomes: positives={labels.sum():,}; negatives={(labels == 0).sum():,}; reviewed prevalence={labels.mean():.3%}")
    print("This checks structure and source links only. A human remains responsible for label decisions and complete follow-up review.")


if __name__ == "__main__":
    main()
