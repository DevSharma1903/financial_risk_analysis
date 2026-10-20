"""Create a source-linked manual review queue from same-period amendment candidates.

The queue is for academic label curation. An amendment is only a candidate event;
it must not be treated as a confirmed restatement without evidence review.
"""
from pathlib import Path
import argparse
import pandas as pd

REQUIRED = {
    "adsh_original", "adsh_amendment", "cik", "period", "form_original",
    "filed_original", "filed_amendment", "days_after_original", "amendment_url",
}


def build_review_queue(candidates: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(candidates.columns)
    if missing:
        raise ValueError(f"Candidate file missing columns: {sorted(missing)}")
    out = candidates.copy()
    out["review_status"] = "pending"
    out["event_type"] = ""
    out["related_8k_adsh"] = ""
    out["related_8k_url"] = ""
    out["evidence_excerpt"] = ""
    out["reviewer"] = ""
    out["review_notes"] = ""
    out["is_confirmed_restatement"] = pd.Series(pd.NA, index=out.index, dtype="Int8")
    out = out.rename(columns={
        "adsh_original": "original_adsh",
        "adsh_amendment": "candidate_event_adsh",
        "form_original": "original_form",
        "filed_original": "original_filed",
        "filed_amendment": "candidate_event_filed",
        "days_after_original": "days_to_candidate_event",
        "amendment_url": "candidate_event_url",
    })
    out = out.sort_values(["original_filed", "original_adsh", "candidate_event_filed"])
    return out.reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("data/restatement_candidates.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/restatement_candidate_review.csv"))
    args = parser.parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Candidate file not found: {args.input}")
    candidates = pd.read_csv(args.input, dtype={"cik": str, "adsh_original": str, "adsh_amendment": str})
    review = build_review_queue(candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.output, index=False)
    print(f"Wrote {len(review):,} review candidates to {args.output}")
    print("Review each source filing and supporting Item 4.02 disclosure before assigning a positive label.")
    print("Pending and unclear rows are not verified positives.")


if __name__ == "__main__":
    main()
