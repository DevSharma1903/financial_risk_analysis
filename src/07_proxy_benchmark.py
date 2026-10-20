"""Diagnostic benchmark for the existing amendment proxy, never a restatement score."""

from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def metrics(y, prob, threshold):
    pred = prob >= threshold
    return {
        "n": int(len(y)),
        "prevalence": float(np.mean(y)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "pr_auc": float(average_precision_score(y, prob)),
        "roc_auc": float(roc_auc_score(y, prob)),
        "confusion_matrix": confusion_matrix(y, pred, labels=[0, 1]).tolist(),
    }


def main():
    data = Path(__file__).resolve().parents[1] / "data"
    df = pd.read_csv(data / "mda_corpus.csv", usecols=["adsh", "cik", "filed", "y", "mda_text"], dtype=str)
    df = df[df.mda_text.notna() & df.mda_text.str.strip().ne("")].copy()
    # Keep the diagnostic bounded on long filings; this is not the production extractor.
    df["mda_text"] = df.mda_text.str.slice(0, 20_000)
    df["year"] = df.filed.str[:4].astype(int)
    df["y"] = df.y.astype(int)
    train = df[df.year <= 2021]
    val = df[df.year == 2023]
    test = df[df.year == 2025]
    if any(part.y.nunique() != 2 for part in [train, val, test]):
        raise ValueError("Each chronological split needs both classes")

    vectorizer = TfidfVectorizer(
        strip_accents="unicode", lowercase=True, ngram_range=(1, 2),
        min_df=5, max_df=0.95, max_features=30_000, sublinear_tf=True,
        dtype=np.float32,
    )
    x_train = vectorizer.fit_transform(train.mda_text)
    x_val = vectorizer.transform(val.mda_text)
    x_test = vectorizer.transform(test.mda_text)
    selected = None
    for c in (0.1, 1.0, 10.0):
        model = LogisticRegression(C=c, max_iter=1000, solver="liblinear", random_state=42)
        model.fit(x_train, train.y)
        prob = model.predict_proba(x_val)[:, 1]
        for threshold in np.arange(0.05, 0.951, 0.025):
            score = f1_score(val.y, prob >= threshold)
            candidate = (score, -c, -threshold, model, threshold)
            if selected is None or candidate[:3] > selected[:3]:
                selected = candidate

    model, threshold = selected[3:]
    report = {
        "warning": "Labels mean later amendment, not confirmed restatement. The source corpus was sampled 50/50; these are not natural-prevalence metrics.",
        "train_rows": len(train),
        "validation": metrics(val.y, model.predict_proba(x_val)[:, 1], threshold),
        "test": metrics(test.y, model.predict_proba(x_test)[:, 1], threshold),
        "always_negative_test_accuracy": float(1 - test.y.mean()),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
