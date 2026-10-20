"""Create a review queue of later amendments; these are NOT restatement labels."""

from pathlib import Path
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
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
    matches = matches[matches.days_after_original.between(1, 365)].copy()
    matches = matches.sort_values(["adsh_original", "filed_date_amendment"])
    matches = matches.drop_duplicates("adsh_original", keep="first")
    matches["amendment_url"] = matches.apply(
        lambda row: (
            f"https://www.sec.gov/Archives/edgar/data/{int(row.cik)}/"
            f"{row.adsh_amendment.replace('-', '')}/{row.adsh_amendment}.txt"
        ), axis=1,
    )
    out = root / "data" / "restatement_candidates.csv"
    matches[[
        "adsh_original", "adsh_amendment", "cik", "period", "form_original",
        "filed_original", "filed_amendment", "days_after_original", "amendment_url",
    ]].to_csv(out, index=False)
    print(f"Wrote {len(matches):,} amendment candidates to {out}")
    print("Candidate means later amendment, not confirmed financial restatement.")


if __name__ == "__main__":
    main()
