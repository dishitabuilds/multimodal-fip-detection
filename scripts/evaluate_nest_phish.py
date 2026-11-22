"""Evaluate frozen auxiliary text artifacts on independent matched emails."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

import joblib

from fipd.models.message_text import normalize_message
from train_smishing_text import measures

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/raw/nest_phish/final_email_dataset.csv"
EXPECTED_SHA256 = "6dca04a523126eb08d83378b2de1cd869a721e60c8032d2eaaf118d0f42aee67"
OUTPUT = ROOT / "data/processed/nest_phish_external_evaluation.json"


def main() -> int:
    if hashlib.sha256(SOURCE.read_bytes()).hexdigest() != EXPECTED_SHA256:
        raise ValueError("NEST-Phish file does not match the evaluated release")
    with SOURCE.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 146 or {row["true_label"] for row in rows} != {"legit", "phish"}:
        raise ValueError("unexpected NEST-Phish cohort")
    texts = [normalize_message(row["subject"] + " " + row["body"]) for row in rows]
    labels = [int(row["true_label"] == "phish") for row in rows]
    results = {}
    for name in ("email_text", "smishing_text", "joint_message_text"):
        path = ROOT / "data/processed" / name / "model.joblib"
        model = joblib.load(path)
        if name == "joint_message_text":
            features = model["vectorizer"].transform(texts)
            scores = model["classifier"].predict_proba(features)[:, 1]
        else:
            scores = model.predict_proba(texts)[:, 1]
        by_theme = {}
        for theme in sorted({row["theme"] for row in rows}):
            indices = [i for i, row in enumerate(rows) if row["theme"] == theme]
            by_theme[theme] = measures([labels[i] for i in indices], scores[indices])
        results[name] = {"overall": measures(labels, scores), "by_theme": by_theme,
                         "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {
        "dataset": "NEST-Phish final 146 synthetic matched workplace emails",
        "dataset_url": "https://www.kaggle.com/datasets/emilywinokur/synthetic-email-corpus",
        "dataset_license": "CC BY-SA 4.0 (dataset README)",
        "dataset_sha256": EXPECTED_SHA256,
        "role": "external_evaluation_after_models_frozen; future tuning makes it development data",
        "input": "subject + body text; no sender, QR, or screenshot pixels",
        "threshold": 0.5, "scores_are_calibrated": False,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"dataset": report["dataset"],
                      "results": {name: data["overall"] for name, data in results.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
