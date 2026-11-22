"""Train and source-holdout-test an auxiliary phishing-email text model."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

from fipd.models.message_text import duplicate_key, normalize_message
from train_smishing_text import measures

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/meajor_cleaned_preprocessed.parquet.gzip"
OUTPUT = ROOT / "data/processed/email_text"


def read_email_cohorts():
    frame = pd.read_parquet(SOURCE, columns=["source", "label", "subject", "body"])
    frame = frame.dropna(subset=["source", "label"])
    frame = frame[frame["source"].isin(["trec5", "trec6", "trec7"])]
    frame["text"] = (frame["subject"].fillna("") + " " + frame["body"].fillna(""))
    frame["text"] = frame["text"].map(lambda value: normalize_message(value[:5000]))
    frame["key"] = frame["text"].map(duplicate_key)
    frame = frame[frame["key"] != ""]
    conflicts = frame.groupby("key")["label"].nunique()
    frame = frame[~frame["key"].isin(conflicts[conflicts > 1].index)]
    # A normalized duplicate in any source belongs to only one partition.
    frame["source_rank"] = frame["source"].map({"trec5": 0, "trec6": 1, "trec7": 2})
    frame = frame.sort_values("source_rank").drop_duplicates("key", keep="first")
    cohorts = {source: frame[frame["source"] == source] for source in ("trec5", "trec6", "trec7")}
    if any(set(part["label"].astype(int)) != {0, 1} for part in cohorts.values()):
        raise ValueError("every source split must contain both classes")
    return cohorts


def main() -> int:
    cohorts = read_email_cohorts()
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=100000,
                        min_df=3, sublinear_tf=True),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42),
    )
    train, validation, test = (cohorts[name] for name in ("trec5", "trec6", "trec7"))
    model.fit(train["text"], train["label"].astype(int))
    validation_scores = model.predict_proba(validation["text"])[:, 1]
    test_scores = model.predict_proba(test["text"])[:, 1]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT / "model.joblib"
    joblib.dump(model, artifact, compress=3)
    report = {
        "task": "benign_vs_phishing_email_text_only", "purpose": "auxiliary_research",
        "source": "Cardoso et al., MeAJOR v2, Zenodo DOI 10.5281/zenodo.18471483",
        "source_md5": hashlib.md5(SOURCE.read_bytes()).hexdigest(),
        "split": {"train_trec5": len(train), "validation_trec6": len(validation),
                  "test_trec7": len(test),
                  "method": "source holdout; normalized duplicate and conflicting label removal"},
        "validation": measures(validation["label"].astype(int), validation_scores),
        "test": measures(test["label"].astype(int), test_scores),
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "limitations": ["Email text, not screenshots", "Anonymized older email sources",
                        "No QR destination or verified sender available from OCR"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
