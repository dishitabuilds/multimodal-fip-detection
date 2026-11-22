"""Evaluate frozen text models on a separate template-tagged email cohort."""
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
DATA = ROOT / "data/raw/phishing_validation_emails_v3.csv"
META = ROOT / "data/raw/phishing_v3_metadata.csv"
EXPECTED_MD5 = {DATA.name: "7047687b5a0040907f40b8728cfd3d68",
                META.name: "3b79f7817d7f9f631032689c506b3414"}
OUTPUT = ROOT / "data/processed/validation_emails_v3_external_evaluation.json"


def main() -> int:
    for path in (DATA, META):
        if hashlib.md5(path.read_bytes()).hexdigest() != EXPECTED_MD5[path.name]:
            raise ValueError(f"external dataset file changed: {path.name}")
    with DATA.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    with META.open(encoding="utf-8-sig", newline="") as stream:
        metadata = list(csv.DictReader(stream))
    if len(rows) != len(metadata) or len(rows) != 2000:
        raise ValueError("email and template metadata coverage differs")
    if any(row["Email Type"] != meta["email_type"] or int(meta["record_id"]) != i
           for i, (row, meta) in enumerate(zip(rows, metadata), 1)):
        raise ValueError("email labels do not align with template metadata")
    labels = [int(row["Email Type"] == "Phishing Email") for row in rows]
    texts = [normalize_message(row["Email Text"]) for row in rows]
    results = {}
    for name in ("email_text", "smishing_text", "joint_message_text", "email_nest_augmented"):
        path = ROOT / "data/processed" / name / "model.joblib"
        model = joblib.load(path)
        if isinstance(model, dict):
            scores = model["classifier"].predict_proba(model["vectorizer"].transform(texts))[:, 1]
        else:
            scores = model.predict_proba(texts)[:, 1]
        by_category = {}
        for category in sorted({row["source_category"] for row in metadata}):
            indices = [i for i, row in enumerate(metadata) if row["source_category"] == category]
            by_category[category] = measures([labels[i] for i in indices], scores[indices])
        by_template = {}
        for template in sorted({row["template_id"] for row in metadata}):
            indices = [i for i, row in enumerate(metadata) if row["template_id"] == template]
            by_template[template] = measures([labels[i] for i in indices], scores[indices])
        results[name] = {"overall": measures(labels, scores),
                         "by_category": by_category, "by_template": by_template,
                         "artifact_sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    report = {
        "dataset": "Phishing validation emails v3",
        "dataset_url": "https://zenodo.org/records/23954793",
        "dataset_md5": EXPECTED_MD5,
        "role": "independent_external_stress_set_after_model_selection",
        "note": "95% template-generated; 100 collected messages; text only",
        "threshold": 0.5, "scores_are_calibrated": False,
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({name: result["overall"] for name, result in results.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
