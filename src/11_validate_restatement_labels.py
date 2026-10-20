"""Validate a manually adjudicated restatement event ledger.

This script checks completeness and provenance fields only. It does not infer
labels from amendments, generate negatives, or decide whether an event qualifies.
Run locally before uploading the ledger to Colab for GPU training.
"""
from pathlib import Path
import argparse
import pandas as pd


REQUIRED = {
    "original_adsh", "original_filed_date", "label", "review_status", "event_type", "event_date",
    "evidence_url", "evidence_excerpt", "reviewer", "review_notes",
}
ALLOWED_EVENT_TYPES = {
    "restatement", "revision_not_restatement", "other_amendment",
    "no_confirmed_restatement",
}


def validate_ledger(frame: pd.DataFrame) -> list[str]:
    errors = []
    missing = REQUIRED - set(frame.columns)
    if missing:
        return [f"Missing required columns: {sorted(missing)}"]
    if frame.original_adsh.isna().any() or frame.original_adsh.astype(str).str.strip().eq("").any():
        errors.append("original_adsh must be present on every adjudicated row")
    if frame.original_adsh.duplicated().any():
        errors.append("original_adsh must be unique (one final adjudication per filing)")

    labels = pd.to_numeric(frame.label, errors="coerce")
    if labels.isna().any() or not labels.isin([0, 1]).all():
        errors.append("label must be 0 or 1 for every row; pending/unknown rows are not training labels")
    if not frame.review_status.astype(str).str.lower().eq("reviewed").all():
        errors.append("Every included row must have review_status=reviewed")
    event_types = frame.event_type.fillna("").astype(str).str.strip()
    if not event_types.isin(ALLOWED_EVENT_TYPES).all():
        errors.append(f"event_type must be one of {sorted(ALLOWED_EVENT_TYPES)}")
    expected_label = event_types.eq("restatement").astype(int)
    if labels.notna().all() and not labels.astype(int).eq(expected_label).all():
        errors.append("label must be 1 only for event_type=restatement and 0 for the other reviewed types")

    event_dates = pd.to_datetime(frame.event_date, errors="coerce")
    original_dates = pd.to_datetime(frame.original_filed_date, errors="coerce")
    positive = labels.eq(1)
    if (positive & event_dates.isna()).any():
        errors.append("Positive restatement rows need an event_date")
    if original_dates.isna().any():
        errors.append("original_filed_date must be present and parseable on every row")
    for column in ("evidence_url", "evidence_excerpt", "reviewer"):
        values = frame[column].fillna("").astype(str).str.strip()
        if values.eq("").any():
            errors.append(f"Every reviewed row needs {column}")
    urls = frame.evidence_url.fillna("").astype(str).str.lower()
    if (~urls.str.startswith("https://www.sec.gov/")).any():
        errors.append("Evidence URLs must point to an SEC.gov primary filing source")
    if (positive & original_dates.isna()).any():
        errors.append("original_filed_date must be present on positive rows")
    if (positive & event_dates.lt(original_dates)).any():
        errors.append("Positive event_date cannot precede the original filing date")
    if (positive & event_dates.gt(original_dates + pd.Timedelta(days=365))).any():
        errors.append("Positive event_date must fall within 365 days of the original filing")
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ledger", type=Path, help="Manually reviewed event ledger CSV")
    args = parser.parse_args()
    if not args.ledger.is_file():
        raise FileNotFoundError(args.ledger)
    frame = pd.read_csv(args.ledger, dtype={"original_adsh": str})
    errors = validate_ledger(frame)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        raise SystemExit(1)
    labels = pd.to_numeric(frame.label).astype(int)
    print(f"Valid reviewed ledger: {len(frame):,} filings; positives={labels.sum():,}; prevalence={labels.mean():.3%}")
    print("Validation checks structure and evidence fields; a human remains responsible for label correctness and complete negative review.")


if __name__ == "__main__":
    main()
