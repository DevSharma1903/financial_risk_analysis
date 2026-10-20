"""Read-only audit of the local SEC data and cached training artifacts."""

from pathlib import Path
import json
import numpy as np
import pandas as pd


def main():
    root = Path(__file__).resolve().parents[1]
    data = root / "data"
    index = pd.read_csv(data / "labeled_index.csv", usecols=["adsh", "cik", "period", "filed", "y"], dtype=str)
    corpus = pd.read_csv(data / "mda_corpus.csv", usecols=["adsh", "cik", "period", "filed", "y", "mda_text"], dtype=str)
    valid = corpus.mda_text.notna() & corpus.mda_text.str.strip().ne("")
    x = np.load(data / "X.npy", mmap_mode="r")
    y = np.load(data / "y.npy", mmap_mode="r")

    report = {
        "index_rows": len(index),
        "index_class_counts": index.y.value_counts().to_dict(),
        "corpus_rows": len(corpus),
        "corpus_valid_text_rows": int(valid.sum()),
        "corpus_duplicate_accessions": int(corpus.adsh.duplicated().sum()),
        "corpus_duplicate_company_periods": int(corpus.duplicated(["cik", "period"]).sum()),
        "cached_x_shape": list(x.shape),
        "cached_y_shape": list(y.shape),
        "cached_class_counts": dict(zip(*[a.tolist() for a in np.unique(y, return_counts=True)])),
        "cache_matches_current_corpus": bool(len(x) == len(y) == valid.sum()),
        "cache_matches_current_feature_width": bool(x.shape[1] == 1566),
        "corpus_years": corpus.loc[valid, "filed"].str[:4].value_counts().sort_index().to_dict(),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
