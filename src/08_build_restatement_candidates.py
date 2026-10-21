"""Create a review queue of same-period amendments; these are NOT labels.

The default five-year window supplies candidate events for the restatement
review study. The event window is only a retrieval aid; reviewed labels use
the actual documented event date and explicit follow-up horizon.
"""

from pathlib import Path
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--max-years', type=int, default=5,
                        help='Calendar-year candidate retrieval window; 0 means no limit (default: 5).')
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    if args.max_years < 0:
        parser.error('--max-years must be 0 or a positive integer')
    root = args.root.resolve()
    files = sorted((root / "data" / "raw").glob("*/sub.txt"))
    if not files:
        raise FileNotFoundError("No SEC sub.txt files found")
    parts = []
    for path in files:
        part = pd.read_csv(path, sep="\t", dtype=str, usecols=["adsh", "cik", "form", "period", "filed"])
        parts.append(part[part.form.isin(["10-K", "10-Q", "10-K/A", "10-Q/A"])])
    filings = pd.concat(parts, ignore_index=True).drop_duplicates("adsh")
    filings["filed_date"] = pd.to_datetime(filings.filed, format="%Y%m%d", errors="coerce")
    originals = filings[filings.form.isin(["10-K", "10-Q"])].copy()
    amendments = filings[filings.form.isin(["10-K/A", "10-Q/A"])].copy()
    amendments["original_form"] = amendments.form.str.removesuffix("/A")
    matches = originals.merge(
        amendments,
        left_on=["cik", "period", "form"],
        right_on=["cik", "period", "original_form"],
        suffixes=("_original", "_amendment"),
    )
    matches["days_after_original"] = (matches.filed_date_amendment - matches.filed_date_original).dt.days
    matches = matches[matches.days_after_original.ge(1)].copy()
    if args.max_years:
        horizon = matches.filed_date_original + pd.DateOffset(years=args.max_years)
        matches = matches[matches.filed_date_amendment.le(horizon)].copy()
    matches = matches.sort_values(["adsh_original", "filed_date_amendment"])
    matches = matches.drop_duplicates("adsh_original", keep="first")
    matches["amendment_url"] = matches.apply(
        lambda row: (
            f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/"
            f"{row.adsh_amendment.replace('-', '')}/{row.adsh_amendment}.txt"
        ), axis=1,
    )
    out = (args.output.resolve() if args.output else root / "data" / "restatement_candidates.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    matches[[
        "adsh_original", "adsh_amendment", "cik", "period", "form_original",
        "filed_original", "filed_amendment", "days_after_original", "amendment_url",
    ]].to_csv(out, index=False)
    print(f"Wrote {len(matches):,} amendment candidates to {out}")
    print(f"Retrieval window: {'unlimited' if not args.max_years else f'{args.max_years:,} calendar years'}.")
    print("Candidate means later amendment, not confirmed financial restatement.")


if __name__ == "__main__":
    main()
