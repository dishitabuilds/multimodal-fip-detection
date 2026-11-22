"""Test matched enterprise-email augmentation with held-out themes."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from fipd.models.message_text import duplicate_key, normalize_message
from train_email_text import read_email_cohorts
from train_smishing_text import measures

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/nest_phish/final_email_dataset.csv"
EXPECTED_SHA256 = "6dca04a523126eb08d83378b2de1cd869a721e60c8032d2eaaf118d0f42aee67"
OUTPUT = ROOT / "data/processed/email_nest_augmented"
VALIDATION_THEMES = {
    "Account/security notifications (IT/auth)",
    "IT maintenance / access continuity",
    "Operations/field/grid dispatch",
}
TEST_THEMES = {
    "Document sharing/collaboration prompts",
    "Finance/AP/vendor billing",
    "Shipping/delivery notifications",
    "Travel/calendar logistics",
}


def nest_cohorts():
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("NEST-Phish dataset changed")
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 146 or {row["true_label"] for row in rows} != {"legit", "phish"}:
        raise ValueError("unexpected NEST-Phish cohort")
    result = {"train": [], "validation": [], "test": []}
    seen = {}
    for row in rows:
        theme = row["theme"]
        split = "test" if theme in TEST_THEMES else "validation" if theme in VALIDATION_THEMES else "train"
        text = normalize_message(row["subject"] + " " + row["body"])
        key = duplicate_key(text)
        if not key:
            raise ValueError("empty NEST-Phish message")
        if key in seen:
            if seen[key] != (split, row["true_label"]):
                raise ValueError("duplicate NEST-Phish text crosses split or label")
            continue
        seen[key] = (split, row["true_label"])
        result[split].append((text, int(row["true_label"] == "phish"), theme))
    if any({label for _, label, _ in group} != {0, 1} for group in result.values()):
        raise ValueError("every NEST-Phish split must contain both classes")
    return result


def main() -> int:
    trec = read_email_cohorts()
    nest = nest_cohorts()
    train_text = list(trec["trec5"]["text"]) + [row[0] for row in nest["train"]]
    train_label = np.array(list(trec["trec5"]["label"].astype(int)) +
                           [row[1] for row in nest["train"]])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000,
                                 min_df=3, sublinear_tf=True)
    train_features = vectorizer.fit_transform(train_text)
    features = {
        "trec_validation": vectorizer.transform(trec["trec6"]["text"]),
        "trec_test": vectorizer.transform(trec["trec7"]["text"]),
        "nest_validation": vectorizer.transform([row[0] for row in nest["validation"]]),
        "nest_test": vectorizer.transform([row[0] for row in nest["test"]]),
    }
    truth = {
        "trec_validation": list(trec["trec6"]["label"].astype(int)),
        "trec_test": list(trec["trec7"]["label"].astype(int)),
        "nest_validation": [row[1] for row in nest["validation"]],
        "nest_test": [row[1] for row in nest["test"]],
    }
    candidates = {}
    for weight in (1.0, 25.0, 100.0):
        sample_weights = np.ones(len(train_label))
        sample_weights[len(trec["trec5"]):] = weight
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000,
                                        random_state=42)
        classifier.fit(train_features, train_label, sample_weight=sample_weights)
        metrics = {name: measures(truth[name], classifier.predict_proba(features[name])[:, 1])
                   for name in ("trec_validation", "nest_validation")}
        candidates[weight] = (classifier, metrics)

    def validation_score(weight):
        values = candidates[weight][1].values()
        return min(item["macro_f1"] - (item["real_false_positive_rate"] or 0.0) for item in values)

    selected_weight = max(candidates, key=validation_score)
    classifier = candidates[selected_weight][0]
    test = {name: measures(truth[name], classifier.predict_proba(features[name])[:, 1])
            for name in ("trec_test", "nest_test")}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT / "model.joblib"
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier,
                 "task": "email_with_matched_enterprise_examples",
                 "nest_weight": selected_weight}, artifact, compress=3)
    report = {
        "purpose": "auxiliary_research", "task": "email_with_matched_enterprise_examples",
        "dataset_sha256": EXPECTED_SHA256,
        "nest_themes": {"validation": sorted(VALIDATION_THEMES), "test": sorted(TEST_THEMES)},
        "counts": {name: len(group) for name, group in nest.items()},
        "selection": "validation only; minimum macro-F1 minus false-positive rate across TREC6 and NEST validation",
        "candidate_validation": {str(weight): metrics for weight, (_, metrics) in candidates.items()},
        "selected_nest_weight": selected_weight,
        "test": test,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "limitations": ["NEST is synthetic and small", "NEST themes were viewed during baseline evaluation",
                        "Text only; screenshot OCR and QR destinations excluded"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
