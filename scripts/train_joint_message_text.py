"""Compare joint email/SMS text models with per-format, source-separated tests."""
from __future__ import annotations

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
from sklearn.model_selection import train_test_split

from fipd.models.message_text import duplicate_key
from train_email_text import read_email_cohorts
from train_smishing_text import SOURCE as SMS_SOURCE, read_training, measures

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data/processed/joint_message_text"


def cohorts():
    email = read_email_cohorts()
    sms_rows = read_training(SMS_SOURCE)
    sms_labels = np.array([label for _, label in sms_rows])
    sms_train, remainder = train_test_split(np.arange(len(sms_rows)), test_size=0.3,
                                             random_state=42, stratify=sms_labels)
    sms_val, sms_test = train_test_split(remainder, test_size=0.5,
                                         random_state=42, stratify=sms_labels[remainder])

    def sms(indices):
        return [(sms_rows[i][0], sms_rows[i][1]) for i in indices]

    result = {
        "train": {"email": list(zip(email["trec5"]["text"], email["trec5"]["label"].astype(int))),
                  "sms": sms(sms_train)},
        "validation": {"email": list(zip(email["trec6"]["text"], email["trec6"]["label"].astype(int))),
                       "sms": sms(sms_val)},
        "test": {"email": list(zip(email["trec7"]["text"], email["trec7"]["label"].astype(int))),
                 "sms": sms(sms_test)},
    }
    # Any normalized message shared across formats or sources is owned by the
    # earliest split. Exact or numeric-template overlap cannot reach a test.
    seen = set()
    for split in ("train", "validation", "test"):
        for format_name in ("email", "sms"):
            kept = []
            for text, label in result[split][format_name]:
                key = duplicate_key(text)
                if key not in seen:
                    kept.append((text, label))
                    seen.add(key)
            result[split][format_name] = kept
            if {label for _, label in kept} != {0, 1}:
                raise ValueError(f"{split}/{format_name} lacks both labels")
    return result


def main() -> int:
    groups = cohorts()
    train = groups["train"]
    texts = [text for group in train.values() for text, _ in group]
    labels = np.array([label for group in train.values() for _, label in group])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100000,
                                 min_df=3, sublinear_tf=True)
    train_features = vectorizer.fit_transform(texts)
    features = {split: {name: vectorizer.transform([text for text, _ in rows])
                        for name, rows in formats.items()}
                for split, formats in groups.items() if split != "train"}
    email_count = len(train["email"])
    candidates = {}
    for sms_weight in (1.0, 3.0):
        weights = np.ones(len(labels))
        weights[email_count:] = sms_weight
        classifier = LogisticRegression(class_weight="balanced", max_iter=1000,
                                        random_state=42)
        classifier.fit(train_features, labels, sample_weight=weights)
        validation = {}
        for name, rows in groups["validation"].items():
            scores = classifier.predict_proba(features["validation"][name])[:, 1]
            validation[name] = measures([label for _, label in rows], scores)
        candidates[sms_weight] = (classifier, validation)
    # The score is set before looking at either test cohort. Penalize false
    # positives because a "Fake" verdict on real content is consequential.
    def selection_score(weight):
        per_format = candidates[weight][1]
        return min(item["macro_f1"] - (item["real_false_positive_rate"] or 0.0)
                   for item in per_format.values())

    selected_weight = max(candidates, key=selection_score)
    classifier = candidates[selected_weight][0]
    test = {}
    for name, rows in groups["test"].items():
        scores = classifier.predict_proba(features["test"][name])[:, 1]
        test[name] = measures([label for _, label in rows], scores)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT / "model.joblib"
    joblib.dump({"vectorizer": vectorizer, "classifier": classifier,
                 "task": "joint_email_sms_text", "sms_weight": selected_weight},
                artifact, compress=3)
    report = {
        "purpose": "auxiliary_research", "task": "joint_email_sms_text",
        "counts": {split: {name: len(rows) for name, rows in formats.items()}
                   for split, formats in groups.items()},
        "split_method": "Email TREC5/6/7 source holdout; SMS normalized-template split; cross-format overlap removed",
        "validation_candidates": {str(weight): result for weight, (_, result) in candidates.items()},
        "selected_sms_weight": selected_weight,
        "test": test,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "limitations": ["Text only, not source screenshots", "Email and SMS labels come from different sources",
                        "SMS test is a random split of one source", "OCR and QR evidence excluded"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
