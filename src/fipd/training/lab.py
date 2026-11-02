"""Reproducible classical-model comparisons with validation-only selection."""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import csv
import json
from pathlib import Path
import platform
import re
import time
import uuid

import numpy as np

from ..datasets.examples import INPUT_PIPELINE_VERSION
from ..datasets.experiments import common_cohort, file_hash, load_experiment_inputs
from ..models.classical import (
    LAB_PIPELINE_VERSION, MODEL_NAMES, LabFeatureSpace, MobileNetFeatures,
    build_classifier, classifier_scores, load_lab_artifact,
)
from ..models.config import ARMS, TextConfig
from .metrics import evaluate


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def _settings(config):
    text = TextConfig(**config.get("text", {"fields": ["ocr_text", "claim_text"],
                                          "transliterate": False}))
    if not text.fields or set(text.fields) - {"ocr_text", "claim_text"}:
        raise ValueError("lab text fields may contain only OCR and the original caption")
    return text


def run_lab_experiment(records_file, manifest_file, data_dir, output_root, config,
                       *, arms=None, models=None, internal_pilot=False,
                       experiment_id=None, image_weights=None):
    """Fit on train, select C/alpha on validation, score test once per model.

    Every selected arm uses identical examples. The output includes fitted
    preprocessing, predictions, comparison CSV, confusion counts, provenance,
    timing and validation candidates. Existing experiments are never overwritten.
    """
    import joblib
    import scipy
    import sklearn

    arms = list(arms or config.get("arms", ARMS))
    models = list(models or config.get("models", MODEL_NAMES))
    if (not arms or set(arms) - set(ARMS) or len(set(arms)) != len(arms)
            or not models or set(models) - set(MODEL_NAMES) or len(set(models)) != len(models)):
        raise ValueError("arms/models must be nonempty, known and unique")
    combinations = [(arm, model) for arm in arms for model in models
                    if model != "naive_bayes" or arm == "text_only"]
    if not combinations:
        raise ValueError("Multinomial Naive Bayes supports text_only; select SVM/LR for images")
    numeric = {name: int(config.get(name, default)) for name, default in (
        ("max_features", 10000), ("batch_size", 16), ("threads", 4), ("benchmark_runs", 20))}
    if any(value < 1 for value in numeric.values()):
        raise ValueError("feature, batch, thread and benchmark counts must be positive")
    candidates = {name: [float(v) for v in config.get(name, [0.1, 1.0, 10.0])]
                  for name in ("regularization", "nb_alpha")}
    if any(not values or any(not np.isfinite(v) or v <= 0 for v in values)
           for values in candidates.values()):
        raise ValueError("C and alpha candidate lists must contain finite positive values")
    seed = int(config.get("seed", 42))
    text_cfg = _settings(config)
    data_dir, output_root = Path(data_dir), Path(output_root)
    records_file, manifest_file = Path(records_file), Path(manifest_file)
    purpose = "internal_smoke_only" if internal_pilot else "research"
    splits, manifest = load_experiment_inputs(
        records_file, manifest_file, text_cfg, allow_internal_smoke=internal_pilot,
        require_manual_review=not internal_pilot)
    needs_images = any(arm != "text_only" for arm in arms)
    cohort, excluded, cohort_hash = common_cohort(splits, data_dir, require_image=needs_images)
    experiment_id = experiment_id or (
        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", experiment_id):
        raise ValueError("experiment ID must be a simple directory name up to 96 characters")
    run_dir = output_root / purpose / experiment_id
    if run_dir.exists():
        raise ValueError(f"experiment already exists: {run_dir}; use a new experiment ID")

    encoder = None
    image_features = {name: None for name in cohort}
    image_ms = 0.0
    if needs_images:
        from ..models.classical import DEFAULT_IMAGE_WEIGHTS

        image_weights = Path(image_weights or DEFAULT_IMAGE_WEIGHTS)
        encoder = MobileNetFeatures(image_weights,
                                    numeric["batch_size"], numeric["threads"])
        tick = time.perf_counter()
        for name, examples in cohort.items():
            image_features[name] = encoder.encode_paths([data_dir / ex.image_path for ex in examples])
        image_ms = (time.perf_counter() - tick) * 1000
    texts = {name: [ex.text for ex in examples] for name, examples in cohort.items()}
    labels = {name: np.array([ex.label for ex in examples]) for name, examples in cohort.items()}
    run_dir.mkdir(parents=True)
    (run_dir / "models").mkdir()
    (run_dir / "evaluations").mkdir()
    identity = {
        "schema_version": 1, "status": "running", "purpose": purpose,
        "unvalidated": internal_pilot, "experiment_id": experiment_id,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "pipeline_version": LAB_PIPELINE_VERSION, "input_pipeline_version": INPUT_PIPELINE_VERSION,
        "config": config, "effective_arms": arms, "effective_models": models,
        "seed": seed, "text_config": asdict(text_cfg),
        "records_sha256": file_hash(records_file), "manifest_sha256": file_hash(manifest_file),
        "assignment_hash": manifest["assignment_hash"], "cohort_hash": cohort_hash,
        "cohort_ids": {name: [ex.uid for ex in examples] for name, examples in cohort.items()},
        "counts": {name: dict(Counter(ex.label for ex in examples))
                   for name, examples in cohort.items()}, "excluded_uids": excluded,
        "label_mapping": {"real": 0, "fake": 1},
        "image_encoder": "mobilenet_v3_small/IMAGENET1K_V1" if encoder else None,
        "image_weights_sha256": encoder.weights_sha256 if encoder else None,
        "image_feature_extraction_ms": image_ms,
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "numpy": np.__version__, "scipy": scipy.__version__},
        "notice": ("Engineering pilot on provisional data; these metrics do not establish "
                   "detection accuracy." if internal_pilot else
                   "Held-out measurements on the supplied reviewed dataset; generalization "
                   "depends on coverage and source/campaign independence."),
    }
    write_json(run_dir / "experiment.json", identity)
    rows = []
    try:
        for arm in arms:
            space = LabFeatureSpace(arm, numeric["max_features"])
            tick = time.perf_counter()
            space.fit(texts["train"], image_features["train"])
            features = {name: space.transform(texts[name], image_features[name]) for name in cohort}
            preprocessing_ms = (time.perf_counter() - tick) * 1000
            for _, model_name in [pair for pair in combinations if pair[0] == arm]:
                print(f"Training {arm} / {model_name} ...", flush=True)
                values = candidates["nb_alpha" if model_name == "naive_bayes" else "regularization"]
                best, best_value, best_validation = None, None, -1.0
                validation = []
                tick = time.perf_counter()
                for value in sorted(set(values)):
                    classifier = build_classifier(model_name, value, seed)
                    classifier.fit(features["train"], labels["train"])
                    scores, threshold, kind = classifier_scores(classifier, features["val"])
                    measured = evaluate(labels["val"], scores, threshold).to_dict()
                    validation.append({"value": value, "metrics": measured})
                    if measured["macro_f1"] > best_validation:
                        best, best_value = classifier, value
                        best_validation = measured["macro_f1"]
                fit_ms = (time.perf_counter() - tick) * 1000
                scores, threshold, kind = classifier_scores(best, features["test"])
                if not np.isfinite(scores).all():
                    raise ValueError("classifier produced nonfinite test scores")
                metrics = evaluate(labels["test"], scores, threshold).to_dict()
                artifact = {
                    "schema_version": 1, "pipeline_version": LAB_PIPELINE_VERSION,
                    "input_pipeline_version": INPUT_PIPELINE_VERSION, "purpose": purpose,
                    "experiment_id": experiment_id, "arm": arm, "model": model_name,
                    "label_mapping": {"real": 0, "fake": 1}, "cohort_hash": cohort_hash,
                    "text_config": asdict(text_cfg), "features": space, "classifier": best,
                    "threshold": threshold, "score_kind": kind, "selected_value": best_value,
                    "image_weights_sha256": encoder.weights_sha256 if arm != "text_only" else None,
                    "sklearn_version": sklearn.__version__,
                }
                key = f"{arm}_{model_name}"
                artifact_path = run_dir / "models" / f"{key}.joblib"
                joblib.dump(artifact, artifact_path, compress=3)
                restored = load_lab_artifact(artifact_path, allow_internal=internal_pilot)
                restored_features = restored["features"].transform(texts["test"], image_features["test"])
                reloaded_scores, _, _ = classifier_scores(restored["classifier"], restored_features)
                np.testing.assert_allclose(reloaded_scores, scores, rtol=1e-7, atol=1e-8)
                # Timed sample includes fresh text/image features and classifier;
                # OCR, file decoding and model initialization are excluded.
                image = None
                if arm != "text_only":
                    from PIL import Image
                    from ..models.classical import canonical_image

                    with Image.open(data_dir / cohort["test"][0].image_path) as source:
                        image = canonical_image(source)

                def infer_one():
                    visual = encoder.encode_images([image]) if image is not None else None
                    sample = space.transform(texts["test"][:1], visual)
                    classifier_scores(best, sample)

                infer_one()
                timings = []
                for _ in range(numeric["benchmark_runs"]):
                    tick = time.perf_counter()
                    infer_one()
                    timings.append((time.perf_counter() - tick) * 1000)
                predictions = [{"uid": ex.uid, "label": ex.label, "score": float(score),
                                "prediction": int(score >= threshold)}
                               for ex, score in zip(cohort["test"], scores)]
                row = {"arm": arm, "model": model_name, "selected_value": best_value,
                       "validation_macro_f1": best_validation, **{k: metrics[k] for k in (
                           "precision", "recall", "f1", "macro_f1", "pr_auc", "roc_auc", "accuracy")},
                       "test_n": metrics["n"], "feature_count": features["train"].shape[1],
                       "fit_ms": fit_ms, "preprocessing_fit_ms": preprocessing_ms,
                       "latency_p50_ms": float(np.percentile(timings, 50)),
                       "latency_p95_ms": float(np.percentile(timings, 95)),
                       "artifact_mb": artifact_path.stat().st_size / 1024**2,
                       "image_weights_mb": (Path(image_weights).stat().st_size / 1024**2
                                            if arm != "text_only" and Path(image_weights).is_file() else 0.0),
                       "score_kind": kind, "artifact": str(artifact_path.resolve())}
                row["deployment_mb"] = row["artifact_mb"] + row["image_weights_mb"]
                write_json(run_dir / "evaluations" / f"{key}.json", {
                    "purpose": purpose, "metrics": metrics, "summary": row,
                    "validation_candidates": validation, "predictions": predictions,
                    "artifact_sha256": file_hash(artifact_path), "reload_verified": True,
                    "benchmark": {"samples": 1, "runs": numeric["benchmark_runs"],
                                  "torch_threads": numeric["threads"] if encoder else None,
                                  "device": "cpu",
                                  "includes": "feature extraction + classifier",
                                  "excludes": "OCR, file decoding, model loading"},
                })
                rows.append(row)
        with (run_dir / "comparison.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        # Choose a recommended model from validation only; never sort by test F1.
        selected = max(rows, key=lambda row: row["validation_macro_f1"])
        identity.update(status="complete", results=rows,
                        recommended_artifact=selected["artifact"],
                        selection_rule="highest validation macro-F1; ties follow configured order")
    except Exception as exc:
        identity.update(status="failed", error=str(exc), results=rows)
        raise
    finally:
        write_json(run_dir / "experiment.json", identity)
    return run_dir, identity
