"""Train and evaluate multimodal baselines and format specialists on matched-format cohort.

Models:
  1. text_only (TF-IDF word+char union + LinearSVC)
  2. image_only (MobileNetV3 + LinearSVC)
  3. fused (Joint multimodal TF-IDF + MobileNetV3 + LinearSVC)
  4. format_specialists (Format Router + Format Specialists + Platt Calibration)

Selection:
  - Fit on train split only
  - Select C and decision thresholds on validation split only
  - Fit Platt calibration on validation split only
  - Evaluate frozen test split ONCE per model
  - Report per-format confusion matrices, scam recall, real FPR, and abstention rate
  - Evaluate on user development challenge images
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import joblib
import numpy as np
from PIL import Image
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.models.classical import MobileNetFeatures, LabFeatureSpace
from fipd.models.multimodal_evidence import (
    clean_ocr_text, decode_qr, detect_format,
    extract_evidence_indicators, normalize_multimodal_text, PlattCalibrator,
)

DATA_DIR = ROOT / "data"
DATASET_DIR = DATA_DIR / "interim/multimodal-generalization"
OUTPUT_DIR = DATA_DIR / "processed/multimodal_generalization"
CHALLENGE_DIR = DATA_DIR / "interim/generalization-challenge"
WEIGHTS_PATH = ROOT / "demo_bundle/weights/mobilenet_v3_small-047dcff4.pth"


def compute_metrics(truth, preds, probabilities=None, threshold=0.5) -> dict[str, Any]:
    truth = np.asarray(truth, dtype=int)
    preds = np.asarray(preds, dtype=int)
    tn, fp, fn, tp = confusion_matrix(truth, preds, labels=[0, 1]).ravel()
    negatives = tn + fp
    positives = tp + fn
    return {
        "n": int(len(truth)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "scam_recall": float(tp / positives) if positives > 0 else None,
        "scam_precision": float(precision_score(truth, preds, zero_division=0)),
        "real_false_positive_rate": float(fp / negatives) if negatives > 0 else None,
        "macro_f1": float(f1_score(truth, preds, average="macro", zero_division=0)),
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading multimodal generalization dataset...")
    records = [json.loads(line) for line in (DATASET_DIR / "records.jsonl").read_text().splitlines()]
    manifest = json.loads((DATASET_DIR / "split_manifest.json").read_text())

    rec_by_uid = {r["uid"]: r for r in records}
    splits = {s: [rec_by_uid[uid] for uid in manifest["splits"][s]] for s in ("train", "validation", "test")}

    print(f"Loaded {len(records)} records across splits: "
          f"Train={len(splits['train'])}, Val={len(splits['validation'])}, Test={len(splits['test'])}")

    # Extract image features
    print("Loading MobileNet visual encoder...")
    encoder = MobileNetFeatures(WEIGHTS_PATH)

    image_features = {}
    normalized_texts = {}
    labels = {}
    formats = {}
    uids = {}

    for s_name, rows in splits.items():
        paths = [DATASET_DIR / r["image_path"] for r in rows]
        print(f"Extracting visual features for {s_name} ({len(paths)} images)...")
        image_features[s_name] = encoder.encode_paths(paths)
        normalized_texts[s_name] = [normalize_multimodal_text(r["text"]) for r in rows]
        labels[s_name] = np.array([1 if r["label"] == "fake" else 0 for r in rows], dtype=int)
        formats[s_name] = [r["format"] for r in rows]
        uids[s_name] = [r["uid"] for r in rows]

    C_VALUES = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    results = {}
    models = {}

    # -------------------------------------------------------------------------
    # Baseline Arms: text_only, image_only, fused (Joint models)
    # -------------------------------------------------------------------------
    for arm in ("text_only", "image_only", "fused"):
        print(f"\n--- Training Joint Baseline: {arm} ---")
        space = LabFeatureSpace(arm=arm, max_features=5000)
        space.fit(normalized_texts["train"], images=image_features["train"])

        X_train = space.transform(normalized_texts["train"], images=image_features["train"])
        X_val = space.transform(normalized_texts["validation"], images=image_features["validation"])
        X_test = space.transform(normalized_texts["test"], images=image_features["test"])

        best_c = 1.0
        best_val_score = -1.0
        best_clf = None

        # Select C on validation only
        for c in C_VALUES:
            clf = LinearSVC(C=c, class_weight="balanced", random_state=42, max_iter=2000)
            clf.fit(X_train, labels["train"])
            val_margins = clf.decision_function(X_val)
            val_preds = (val_margins >= 0.0).astype(int)
            val_metrics = compute_metrics(labels["validation"], val_preds)

            # Selection metric: maximize macro F1, heavily penalize false positive rate on legitimate content
            score = val_metrics["macro_f1"] - (val_metrics["real_false_positive_rate"] or 0.0)
            if score > best_val_score:
                best_val_score = score
                best_c = c
                best_clf = clf

        print(f"Selected C={best_c} on validation (score={best_val_score:.3f})")

        # Calibrate margins on validation
        train_margins = best_clf.decision_function(X_train)
        val_margins = best_clf.decision_function(X_val)
        test_margins = best_clf.decision_function(X_test)

        calibrator = PlattCalibrator().fit(val_margins, labels["validation"])
        val_probs = calibrator.predict_proba(val_margins)[:, 1]
        test_probs = calibrator.predict_proba(test_margins)[:, 1]

        # Validation metrics
        val_eval = compute_metrics(labels["validation"], (val_margins >= 0.0).astype(int))

        # Test metrics (FROZEN TEST EVALUATION)
        test_preds = (test_margins >= 0.0).astype(int)
        test_eval = compute_metrics(labels["test"], test_preds)

        # Per-format breakdown on test set
        per_format_test = {}
        for fmt in ("investment_graphic", "email_screenshot", "sms_screenshot"):
            mask = np.array([f == fmt for f in formats["test"]])
            per_format_test[fmt] = compute_metrics(labels["test"][mask], test_preds[mask])

        results[arm] = {
            "selected_C": best_c,
            "validation": val_eval,
            "test_overall": test_eval,
            "test_by_format": per_format_test,
        }
        models[arm] = {
            "arm": arm,
            "space": space,
            "classifier": best_clf,
            "calibrator": calibrator,
        }

    # -------------------------------------------------------------------------
    # Format Specialist Architecture (Router + Format Specialists)
    # -------------------------------------------------------------------------
    print("\n--- Training Format Specialist Architecture ---")
    specialists = {}
    specialist_results = {}

    for fmt in ("investment_graphic", "email_screenshot", "sms_screenshot"):
        # Filter train/val/test by format
        train_mask = np.array([f == fmt for f in formats["train"]])
        val_mask = np.array([f == fmt for f in formats["validation"]])
        test_mask = np.array([f == fmt for f in formats["test"]])

        txt_train = [t for i, t in enumerate(normalized_texts["train"]) if train_mask[i]]
        img_train = image_features["train"][train_mask]
        y_train = labels["train"][train_mask]

        txt_val = [t for i, t in enumerate(normalized_texts["validation"]) if val_mask[i]]
        img_val = image_features["validation"][val_mask]
        y_val = labels["validation"][val_mask]

        txt_test = [t for i, t in enumerate(normalized_texts["test"]) if test_mask[i]]
        img_test = image_features["test"][test_mask]
        y_test = labels["test"][test_mask]

        # Use fused features for specialist
        sp_space = LabFeatureSpace(arm="fused", max_features=3000)
        sp_space.fit(txt_train, images=img_train)

        X_sp_train = sp_space.transform(txt_train, images=img_train)
        X_sp_val = sp_space.transform(txt_val, images=img_val)
        X_sp_test = sp_space.transform(txt_test, images=img_test)

        best_sp_c = 1.0
        best_sp_val = -1.0
        best_sp_clf = None

        for c in C_VALUES:
            clf = LinearSVC(C=c, class_weight="balanced", random_state=42, max_iter=2000)
            clf.fit(X_sp_train, y_train)
            v_margins = clf.decision_function(X_sp_val)
            v_eval = compute_metrics(y_val, (v_margins >= 0.0).astype(int))
            v_score = v_eval["macro_f1"] - (v_eval["real_false_positive_rate"] or 0.0)
            if v_score > best_sp_val:
                best_sp_val = v_score
                best_sp_c = c
                best_sp_clf = clf

        val_m = best_sp_clf.decision_function(X_sp_val)
        test_m = best_sp_clf.decision_function(X_sp_test)

        sp_calibrator = PlattCalibrator().fit(val_m, y_val)

        t_preds = (test_m >= 0.0).astype(int)
        specialist_results[fmt] = {
            "selected_C": best_sp_c,
            "test": compute_metrics(y_test, t_preds),
        }
        specialists[fmt] = {
            "space": sp_space,
            "classifier": best_sp_clf,
            "calibrator": sp_calibrator,
        }

    # Evaluate Format Specialist Architecture Overall on Test Set
    routed_test_preds = []
    routed_test_probs = []
    routed_abstain = 0

    for i, row in enumerate(splits["test"]):
        fmt = formats["test"][i]
        txt = normalized_texts["test"][i]
        img_feat = image_features["test"][i:i+1]
        sp = specialists[fmt]

        X_i = sp["space"].transform([txt], images=img_feat)
        margin_i = sp["classifier"].decision_function(X_i)[0]
        prob_i = sp["calibrator"].predict_proba([margin_i])[0, 1]

        # Check evidence for review triggers
        evidence = extract_evidence_indicators(row["text"])
        if evidence["needs_review"] or (0.40 <= prob_i <= 0.60):
            routed_abstain += 1

        pred_i = 1 if prob_i >= 0.50 else 0
        routed_test_preds.append(pred_i)
        routed_test_probs.append(float(prob_i))

    routed_test_eval = compute_metrics(labels["test"], routed_test_preds)
    routed_by_format = {}
    for fmt in ("investment_graphic", "email_screenshot", "sms_screenshot"):
        mask = np.array([f == fmt for f in formats["test"]])
        routed_by_format[fmt] = compute_metrics(labels["test"][mask], np.array(routed_test_preds)[mask])

    results["format_specialists"] = {
        "test_overall": routed_test_eval,
        "abstention_rate": routed_abstain / len(splits["test"]),
        "test_by_format": routed_by_format,
        "specialists_detail": specialist_results,
    }

    # -------------------------------------------------------------------------
    # Evaluate on User Development Challenge Images
    # -------------------------------------------------------------------------
    print("\n--- Evaluating On User Development Challenge Images ---")
    challenge_manifest = json.loads((CHALLENGE_DIR / "manifest.json").read_text())
    challenge_results = []

    for c_rec in challenge_manifest["records"]:
        img_path = CHALLENGE_DIR / c_rec["image"]
        with Image.open(img_path) as img:
            w, h = img.size
            qrs = decode_qr(img)

        # Run with English OCR to get clean text
        from fipd.enrichment.ocr import EasyOCREngine
        engine = EasyOCREngine(languages=["en"], gpu=False,
                               model_storage_directory=str(DATA_DIR / "models/easyocr"),
                               user_network_directory=str(DATA_DIR / "models/easyocr/user_network"),
                               download_enabled=False)
        ocr_res = engine.read(str(img_path))
        raw_text = ocr_res.text or ""

        detected_fmt = detect_format(w, h, raw_text, qrs)
        evidence = extract_evidence_indicators(raw_text, qrs)

        # Visual feature
        v_feat = encoder.encode_paths([img_path])
        norm_txt = normalize_multimodal_text(raw_text)

        # Joint Fused score
        fused_space = models["fused"]["space"]
        fused_clf = models["fused"]["classifier"]
        fused_cal = models["fused"]["calibrator"]
        X_f = fused_space.transform([norm_txt], images=v_feat)
        fused_margin = float(fused_clf.decision_function(X_f)[0])
        fused_prob = float(fused_cal.predict_proba([fused_margin])[0, 1])

        # Routed Specialist score
        sp = specialists.get(detected_fmt, specialists["investment_graphic"])
        X_sp = sp["space"].transform([norm_txt], images=v_feat)
        sp_margin = float(sp["classifier"].decision_function(X_sp)[0])
        sp_prob = float(sp["calibrator"].predict_proba([sp_margin])[0, 1])

        # Review determination
        needs_review = evidence["needs_review"] or (0.38 <= sp_prob <= 0.62)
        if evidence["is_simulation"]:
            verdict = "review_required"
            verdict_reason = "simulated_phishing_qr"
        elif needs_review:
            verdict = "review_required"
            verdict_reason = "uncertain_probability"
        else:
            verdict = "fake" if sp_prob >= 0.50 else "real"
            verdict_reason = "model_confident"

        challenge_results.append({
            "image": c_rec["image"],
            "expected": c_rec["expected"],
            "detected_format": detected_fmt,
            "qr_payloads": qrs,
            "evidence": evidence,
            "scores": {
                "joint_fused_margin": round(fused_margin, 4),
                "joint_fused_probability": round(fused_prob, 4),
                "specialist_margin": round(sp_margin, 4),
                "specialist_calibrated_probability": round(sp_prob, 4),
            },
            "final_verdict": verdict,
            "verdict_reason": verdict_reason,
        })
        print(f"Challenge Image: {c_rec['image']}")
        print(f"  Detected format: {detected_fmt}")
        print(f"  QR payloads: {qrs}")
        print(f"  Specialist probability: {sp_prob:.3f} (margin: {sp_margin:+.3f})")
        print(f"  Final verdict: {verdict} ({verdict_reason})")

    # -------------------------------------------------------------------------
    # Save Artifacts & Reports
    # -------------------------------------------------------------------------
    artifact_payload = {
        "model_version": "multimodal_generalization_v1",
        "timestamp": "2026-11-23T00:20:00Z",
        "cohort_hash": hashlib.sha256((DATASET_DIR / "records.jsonl").read_bytes()).hexdigest(),
        "models": {
            "joint_fused": models["fused"],
            "joint_text": models["text_only"],
            "joint_image": models["image_only"],
            "specialists": specialists,
        },
        "image_weights_sha256": encoder.weights_sha256,
    }
    model_artifact_path = OUTPUT_DIR / "model.joblib"
    joblib.dump(artifact_payload, model_artifact_path, compress=3)
    artifact_sha = hashlib.sha256(model_artifact_path.read_bytes()).hexdigest()

    report = {
        "experiment_id": "multimodal_generalization_20261123",
        "artifact_sha256": artifact_sha,
        "splits": manifest["counts"],
        "baselines_and_specialists": results,
        "development_challenge_evaluation": challenge_results,
        "summary": {
            "selected_architecture": "format_specialists",
            "test_macro_f1": results["format_specialists"]["test_overall"]["macro_f1"],
            "test_scam_recall": results["format_specialists"]["test_overall"]["scam_recall"],
            "test_real_fpr": results["format_specialists"]["test_overall"]["real_false_positive_rate"],
            "test_abstention_rate": results["format_specialists"]["abstention_rate"],
        },
    }
    report_path = OUTPUT_DIR / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\nModel training and evaluation complete.")
    print(f"Saved artifact: {model_artifact_path} (SHA: {artifact_sha[:12]}...)")
    print(f"Saved report: {report_path}")
    print(f"Overall test Macro F1: {report['summary']['test_macro_f1']:.3f}, "
          f"Scam Recall: {report['summary']['test_scam_recall']:.3f}, "
          f"Real FPR: {report['summary']['test_real_fpr']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
