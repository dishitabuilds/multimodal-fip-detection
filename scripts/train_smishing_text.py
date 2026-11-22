"""Train an auxiliary ham-vs-smishing text model; no image or email claims."""
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
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from fipd.models.message_text import duplicate_key, normalize_message

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/sms_phishing_mendeley_mirror.csv"
EXTERNAL = ROOT / "data/raw/ncsu_phishing_messages.csv"
OUTPUT = ROOT / "data/processed/smishing_text"


def read_training(path: Path):
    """Keep only ham and smishing, and fail on conflicting duplicate labels."""
    rows = {}
    with path.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
        reader = csv.DictReader(stream)
        if not {"LABEL", "TEXT"}.issubset(reader.fieldnames or []):
            raise ValueError("expected LABEL and TEXT columns")
        for row in reader:
            label = row["LABEL"].lower().strip()
            if label not in {"ham", "smishing"}:
                continue  # Spam is not a phishing ground truth.
            key = duplicate_key(row["TEXT"])
            if not key:
                continue
            value = (normalize_message(row["TEXT"]), int(label == "smishing"))
            if key in rows and rows[key][1] != value[1]:
                raise ValueError("conflicting labels for duplicate message")
            rows.setdefault(key, value)
    return list(rows.values())


def measures(truth, scores, threshold=0.5):
    predicted = np.asarray(scores) >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, predicted, labels=[0, 1]).ravel()
    negatives = tn + fp
    return {"n": int(len(truth)), "tn": int(tn), "fp": int(fp),
            "fn": int(fn), "tp": int(tp),
            "fake_precision": float(precision_score(truth, predicted, zero_division=0)),
            "fake_recall": float(recall_score(truth, predicted, zero_division=0)),
            "macro_f1": float(f1_score(truth, predicted, average="macro", zero_division=0)),
            "real_false_positive_rate": float(fp / negatives) if negatives > 0 else None}


def main() -> int:
    samples = read_training(SOURCE)
    if len(samples) < 1000:
        raise ValueError("too few distinct messages for training")
    texts = [text for text, _ in samples]
    labels = np.array([label for _, label in samples])
    train, remainder = train_test_split(np.arange(len(labels)), test_size=0.3,
                                        random_state=42, stratify=labels)
    validation, test = train_test_split(remainder, test_size=0.5,
                                        random_state=42, stratify=labels[remainder])
    candidates = {}
    for analyzer, ngrams in (("word", (1, 2)), ("char_wb", (3, 5))):
        model = make_pipeline(
            TfidfVectorizer(analyzer=analyzer, ngram_range=ngrams,
                            sublinear_tf=True, min_df=2),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=42),
        )
        model.fit([texts[i] for i in train], labels[train])
        scores = model.predict_proba([texts[i] for i in validation])[:, 1]
        candidates[analyzer] = (model, measures(labels[validation], scores))
    selected_name = max(candidates, key=lambda name: candidates[name][1]["macro_f1"])
    selected = candidates[selected_name][0]
    test_scores = selected.predict_proba([texts[i] for i in test])[:, 1]
    external_recall = None
    if EXTERNAL.is_file():
        with EXTERNAL.open(encoding="utf-8-sig", errors="replace", newline="") as stream:
            external = [normalize_message(row["message"]) for row in csv.DictReader(stream)
                        if row.get("message")]
        external_scores = selected.predict_proba(external)[:, 1]
        external_recall = {"n": len(external), "positive_rate": float((external_scores >= 0.5).mean()),
                           "note": "Reported phishing corpus includes marketing; no benign controls"}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    artifact = OUTPUT / "model.joblib"
    joblib.dump(selected, artifact, compress=3)
    report = {
        "task": "ham_vs_smishing_text_only", "purpose": "auxiliary_research",
        "source": "Mishra and Soni 2022, Mendeley Data DOI 10.17632/f45bkkt8pr.1; GitHub mirror",
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "external_source": "Nahapetyan et al. 2024, NCSU SMS phishing messages",
        "external_sha256": hashlib.sha256(EXTERNAL.read_bytes()).hexdigest() if EXTERNAL.is_file() else None,
        "split": {"train": len(train), "validation": len(validation), "test": len(test),
                  "method": "exact normalized text deduplication before stratified random split"},
        "candidate_validation": {name: metrics for name, (_, metrics) in candidates.items()},
        "selected": selected_name, "test": measures(labels[test], test_scores),
        "external_positive_only": external_recall,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "limitations": ["Text messages, not original screenshots", "Single-source random test",
                        "No independent benign external cohort", "OCR errors and email coverage untested"],
    }
    (OUTPUT / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
