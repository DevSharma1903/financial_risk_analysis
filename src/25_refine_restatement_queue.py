"""Conservatively demote obvious non-event hits from the AI-assisted ledger.

This step keeps ambiguous evidence pending. The output remains a preliminary
issuer-level screen: a later restatement need not cover the exact period of the
original MD&A filing. A human audit is required for confirmatory research.
"""
from pathlib import Path
import argparse
import re

import pandas as pd


NONQUALIFYING = re.compile(
    r"in the event of|potential(?:ly)?\s+(?:future\s+)?restatement|"
    r"risk of (?:a )?restatement|restatement.{0,100}(?:could|would|may|might)\s+(?:be\s+)?prevented|"
    r"could\s+be\s+prevented\s+or\s+detected|articles of incorporation|"
    r"restated certificate of incorporation|restated articles|amending and restating items|"
    r"restated part iv|restated (?:and re-executed )?certifications|"
    r"reverse stock split|reverse recapitalization|new accounting guidance|"
    r"adoption of new accounting guidance|monetary restatement|functional currency|"
    r"did not result in the restatement|did not result in a restatement|no restatement|"
    r"no need for any future restatement|not anticipate the need for any future restatement|"
    r"restatement of (?:a )?merger agreement|merger agreement.{0,100}restated|"
    r"restated to reflect.*(?:stock split|par value)|restated.*discontinued operations",
    re.I,
)


def refine(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    required = {"review_status", "label", "evidence_excerpt", "review_notes"}
    if not required.issubset(frame.columns):
        raise ValueError(f"AI-assisted ledger is missing columns: {sorted(required - set(frame.columns))}")
    out = frame.copy()
    excerpt = out.evidence_excerpt.fillna("").astype(str)
    demote = out.review_status.astype(str).eq("reviewed") & out.label.astype(str).eq("1") & excerpt.str.contains(NONQUALIFYING, na=False)
    out.loc[demote, "review_status"] = "pending"
    for col in ("label", "event_type", "event_date", "outcome_followup_through"):
        if col in out:
            out.loc[demote, col] = ""
    out.loc[demote, "review_notes"] = (
        "AI-assisted screen found a hypothetical disclosure, non-accounting restatement, "
        "presentation change, or otherwise non-confirming phrase. Left pending for human adjudication."
    )
    reviewed = out.review_status.astype(str).eq("reviewed")
    for idx in out.index[reviewed]:
        note = str(out.at[idx, "review_notes"]).rstrip()
        scope = (
            "Outcome scope: issuer-level confirmed restatement reported within the selected horizon; "
            "the label does not establish that the restated period is the original filing period."
        )
        if scope not in note:
            out.at[idx, "review_notes"] = f"{note} {scope}".strip()
    return out, int(demote.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("data/restatement_review_queue_ai_assisted.csv"))
    parser.add_argument("--output", type=Path, default=Path("data/restatement_review_queue_ai_refined.csv"))
    args = parser.parse_args()
    frame = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    out, count = refine(frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"Demoted {count} obvious/non-confirming positive hits to pending.")
    print(f"Wrote {len(out):,} rows to {args.output}")
    print("Statuses:", out.review_status.value_counts().to_dict())
    print("Reviewed labels:", out.loc[out.review_status.eq("reviewed"), "label"].value_counts().to_dict())
    print("This is an AI-assisted exploratory label set, not a human-adjudicated ground truth.")


if __name__ == "__main__":
    main()
