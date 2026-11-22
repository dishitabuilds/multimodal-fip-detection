"""Evaluate local development examples without treating them as a held-out test."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from PIL import Image
from fipd.frontend.analysis import analyze_image
from fipd.models.message_text import normalize_message


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    directory = root / "data/interim/generalization-challenge"
    manifest = json.loads((directory / "manifest.json").read_text())
    auxiliary = {}
    for kind in ("smishing_text", "email_text", "joint_message_text"):
        path = root / "data/processed" / kind / "model.joblib"
        if path.is_file():
            import joblib
            auxiliary[kind] = joblib.load(path)
    outcomes = []
    for record in manifest["records"]:
        path = directory / record["image"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"challenge image changed: {record['image']}")
        with Image.open(path) as image:
            result = analyze_image(image, mode="svm", root=root)
        text = normalize_message(result["raw_ocr"])
        auxiliary_scores = {}
        for name, model in auxiliary.items():
            if not text:
                continue
            if name == "joint_message_text":
                features = model["vectorizer"].transform([text])
                score = model["classifier"].predict_proba(features)[0, 1]
            else:
                score = model.predict_proba([text])[0, 1]
            auxiliary_scores[name] = float(score)
        outcomes.append({"image": record["image"], "expected": record["expected"],
                         "status": result["status"], "label": result["label"],
                         "model_label": result["model_label"],
                         "decision_score": result["decision_score"],
                         "review_reason": result["review_reason"],
                         "detected_format": result.get("detected_format"),
                         "qr_payloads": result.get("qr_payloads", []),
                         "multimodal_assessment": result.get("multimodal_assessment"),
                         "auxiliary_text_scores": auxiliary_scores,
                         "error": result["error"]})
    payload = {"role": "development_challenge_only", "results": outcomes}
    (directory / "latest_result.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))
    return 0 if all(row["status"] == "complete" for row in outcomes) else 1


if __name__ == "__main__":
    raise SystemExit(main())
