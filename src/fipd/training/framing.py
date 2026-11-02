"""Train text framing labels from attributed human annotations.

This task is separate from factual correctness, fraud and investment suitability.
No rule-generated labels, future returns or fact-check verdicts enter its inputs.
Only locally generated joblib artifacts should be loaded (joblib uses pickle).
"""
from __future__ import annotations

from collections import Counter
from datetime import date, datetime, timezone
import hashlib
import ipaddress
import json
from pathlib import Path
import pickle
import re
import struct
import unicodedata
from urllib.parse import urlsplit
import uuid
import warnings

import numpy as np

from ..models.classical import LabFeatureSpace

FRAMING_LABELS = ("return_promise", "urgency", "social_proof", "authority_cue",
                  "selective_context")
FRAMING_PIPELINE_VERSION = "reviewed-framing-tfidf-ovr-svm-v1"
FRAMING_INPUT_VERSION = "ocr-caption-only-v1"
MAX_TEXT = 50000
NOTICE = ("Framing labels are model predictions, not fact checks or findings of fraud. "
          "Signed margins are not probabilities. Selective context requires external "
          "evidence and human review; this text model cannot establish an omission.")


def _text(value, name, *, required=False, limit=MAX_TEXT):
    if not isinstance(value, str) or len(value) > limit or (required and not value.strip()):
        raise ValueError(f"{name} must be {'nonempty ' if required else ''}text up to {limit} characters")
    return value


def _iso_date(value, name):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError(f"{name} must be an ISO date YYYY-MM-DD")
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a valid ISO date YYYY-MM-DD") from exc


def _labels(labels):
    values = list(FRAMING_LABELS if labels is None else labels)
    if (not values or any(not isinstance(x, str) for x in values)
            or len(set(values)) != len(values) or set(values) - set(FRAMING_LABELS)):
        raise ValueError("labels must be a nonempty unique subset of " + ", ".join(FRAMING_LABELS))
    return values


def _source_url(value):
    if (not isinstance(value, str) or len(value) > 2048 or "\\" in value
            or any(character.isspace() or ord(character) < 32 for character in value)):
        raise ValueError("selective_context source_url must be an HTTP or HTTPS URL")
    try:
        parsed = urlsplit(value)
        hostname = (parsed.hostname or "").encode("idna").decode("ascii").rstrip(".")
        valid = (parsed.scheme in {"http", "https"} and hostname
                 and parsed.username is None and parsed.password is None
                 and (parsed.port is None or 1 <= parsed.port <= 65535))
        try:
            valid = valid and ipaddress.ip_address(hostname).is_global
        except ValueError:
            parts = hostname.split(".")
            valid = (valid and len(parts) >= 2 and len(hostname) <= 253
                     and all(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?", part)
                             for part in parts)
                     and not hostname.endswith((".localhost", ".local", ".internal")))
    except (ValueError, UnicodeError):
        valid = False
    if not valid:
        raise ValueError("selective_context source_url must be an HTTP or HTTPS URL")


def framing_text(text, caption=""):
    """Preserve only the promotion's wording, in the same order at inference."""
    _text(text, "text")
    _text(caption, "caption")
    if not text.strip() and not caption.strip():
        raise ValueError("provide the promotion's OCR text or original caption")
    return "\n".join(value.strip() for value in (text, caption) if value.strip())


def _normalized(value):
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def load_framing_records(path, labels=None):
    """Validate attributed annotations, spans and campaign/text split isolation.

    The file records a reviewer's assertions. This does not authenticate identity
    or independently verify the cited external material.
    """
    selected = _labels(labels)
    path = Path(path)
    if not path.is_file():
        raise ValueError(f"No framing annotations at {path}. Create reviewed JSONL following "
                         "docs/FRAMING_LABELS.md; binary labels and rule flags cannot replace review.")
    records, identities, groups, texts = [], set(), {}, {}
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line, parse_constant=lambda x: (_ for _ in ()).throw(
                    ValueError(f"nonfinite JSON value {x}")))
                if not isinstance(row, dict) or row.get("schema_version") != 1:
                    raise ValueError("record must declare schema_version 1")
                for field in ("uid", "group_id"):
                    _text(row.get(field), field, required=True, limit=300)
                uid = row["uid"]
                if uid in identities:
                    raise ValueError(f"duplicate uid {uid}")
                identities.add(uid)
                split = row.get("split")
                if split not in {"train", "val", "test"}:
                    raise ValueError("split must be train, val or test")
                row["text"] = _text(row.get("text", ""), "text")
                row["caption"] = _text(row.get("caption", ""), "caption")
                assembled = framing_text(row["text"], row["caption"])
                as_of = _iso_date(row.get("as_of"), "as_of")
                review = row.get("review")
                if (not isinstance(review, dict) or review.get("method") != "human"
                        or review.get("status") != "approved"):
                    raise ValueError("every record needs an approved human review, including empty labels")
                _text(review.get("reviewer_name"), "reviewer_name", required=True, limit=300)
                reviewed_at = _iso_date(review.get("reviewed_at"), "reviewed_at")
                if reviewed_at < as_of:
                    raise ValueError("reviewed_at cannot precede the promotion as_of date")
                raw_labels = row.get("labels")
                if (not isinstance(raw_labels, list) or any(not isinstance(x, str) for x in raw_labels)
                        or len(set(raw_labels)) != len(raw_labels)
                        or set(raw_labels) - set(FRAMING_LABELS)):
                    raise ValueError("labels must be a list of known unique framing labels")
                evidence = row.get("evidence")
                if not isinstance(evidence, list) or len(evidence) > 100:
                    raise ValueError("evidence must be a list with at most 100 entries")
                covered = set()
                for item in evidence:
                    if not isinstance(item, dict) or item.get("label") not in raw_labels:
                        raise ValueError("each evidence label must be present in labels")
                    field = item.get("field")
                    start, end = item.get("start"), item.get("end")
                    if (field not in {"text", "caption"} or type(start) is not int
                            or type(end) is not int or not 0 <= start < end <= len(row[field])):
                        raise ValueError("evidence needs valid text/caption character start and end offsets")
                    if item.get("quote") != row[field][start:end]:
                        raise ValueError("evidence quote must exactly match the annotated character span")
                    _text(item.get("rationale"), "evidence rationale", required=True, limit=5000)
                    if item["label"] == "selective_context":
                        _source_url(item.get("source_url"))
                        published = _iso_date(item.get("published_at"), "published_at")
                        if published > as_of:
                            raise ValueError("selective_context source published after promotion as_of date")
                        _text(item.get("source_title"), "source_title", required=True, limit=1000)
                        _text(item.get("material_context"), "material_context", required=True, limit=5000)
                    covered.add(item["label"])
                if covered != set(raw_labels):
                    raise ValueError("every positive label needs an exact span and reviewer rationale")
                if row["group_id"] in groups and groups[row["group_id"]] != split:
                    raise ValueError("a campaign/group_id spans multiple splits")
                groups[row["group_id"]] = split
                # Repeated OCR with a changed caption is still the same textual creative.
                keys = {_normalized(assembled)}
                if row["text"].strip():
                    keys.add(_normalized(row["text"]))
                for key in keys:
                    if key in texts and texts[key] != split:
                        raise ValueError("normalized exact promotion text appears across splits")
                    texts[key] = split
                records.append(row)
            except (ValueError, TypeError, KeyError) as exc:
                raise ValueError(f"{path.name} line {number}: {exc}") from exc
    if not records:
        raise ValueError("framing annotation file is empty")
    for split in ("train", "val", "test"):
        split_rows = [row for row in records if row["split"] == split]
        if not split_rows:
            raise ValueError(f"{split} split is empty")
        for label in selected:
            positive = sum(label in row["labels"] for row in split_rows)
            if not 0 < positive < len(split_rows):
                raise ValueError(f"{split} needs both positive and negative examples for {label}")
    return records


def _targets(rows, labels):
    return np.array([[int(label in row["labels"]) for label in labels] for row in rows], dtype=int)


def _scores(classifier, features, n_labels):
    scores = np.asarray(classifier.decision_function(features), dtype=float)
    if scores.ndim == 1:
        scores = scores.reshape(-1, 1)
    if scores.shape != (features.shape[0], n_labels) or not np.isfinite(scores).all():
        raise ValueError("framing classifier returned incompatible or nonfinite margins")
    return scores


def _metrics(truth, predictions, labels):
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, average=None, zero_division=0)
    # A one-column indicator is treated as binary by sklearn; select positive class.
    if len(labels) == 1:
        precision, recall, f1, support = precision[-1:], recall[-1:], f1[-1:], support[-1:]
    result = {"n": len(truth), "macro_f1": float(np.mean(f1)),
            "exact_match": float(accuracy_score(truth, predictions)),
            "per_label": {label: {"precision": float(precision[i]), "recall": float(recall[i]),
                                  "f1": float(f1[i]), "support": int(support[i]),
                                  "negative_support": int(len(truth) - support[i])}
                          for i, label in enumerate(labels)}}
    for i, label in enumerate(labels):
        actual, predicted = truth[:, i].astype(bool), predictions[:, i].astype(bool)
        result["per_label"][label].update({"tp": int(np.sum(actual & predicted)),
                                           "fp": int(np.sum(~actual & predicted)),
                                           "fn": int(np.sum(actual & ~predicted)),
                                           "tn": int(np.sum(~actual & ~predicted))})
    return result


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                          encoding="utf-8")


def run_framing_experiment(records_file, output_root="data/processed/framing", *, labels=None,
                           regularization=(0.1, 1.0, 10.0), max_features=10000,
                           experiment_id=None):
    """Fit train-only TF-IDF and OVR SVM; select C on validation macro F1."""
    import joblib
    import sklearn
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.svm import LinearSVC

    labels = _labels(labels)
    try:
        candidates = [float(value) for value in regularization]
    except (TypeError, ValueError) as exc:
        raise ValueError("C candidates must be finite positive numbers") from exc
    if not candidates or any(not np.isfinite(value) or value <= 0 for value in candidates):
        raise ValueError("C candidates must be finite positive numbers")
    if type(max_features) is not int or not 1 <= max_features <= 1000000:
        raise ValueError("max_features must be an integer from 1 to 1000000")
    records = load_framing_records(records_file, labels)
    splits = {split: [row for row in records if row["split"] == split]
              for split in ("train", "val", "test")}
    texts = {split: [framing_text(row["text"], row["caption"]) for row in rows]
             for split, rows in splits.items()}
    truth = {split: _targets(rows, labels) for split, rows in splits.items()}
    experiment_id = experiment_id or (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-")
                                      + uuid.uuid4().hex[:8])
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", experiment_id):
        raise ValueError("experiment_id must be a simple directory name up to 96 characters")
    run_dir = Path(output_root) / "research" / experiment_id
    if run_dir.exists():
        raise ValueError(f"experiment already exists: {run_dir}; choose a new experiment_id")
    space = LabFeatureSpace("text_only", max_features).fit(texts["train"])
    features = {split: space.transform(values) for split, values in texts.items()}
    candidate_results, best, best_f1, best_c = [], None, -1.0, None
    for value in candidates:
        classifier = OneVsRestClassifier(LinearSVC(C=value, class_weight="balanced", dual="auto",
                                                  random_state=42, max_iter=10000))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                classifier.fit(features["train"], truth["train"])
        except ConvergenceWarning as exc:
            raise ValueError("framing SVM did not converge; review the dataset and C candidates") from exc
        scores = _scores(classifier, features["val"], len(labels))
        metrics = _metrics(truth["val"], (scores >= 0).astype(int), labels)
        candidate_results.append({"C": value, "validation": metrics})
        if metrics["macro_f1"] > best_f1:
            best, best_f1, best_c = classifier, metrics["macro_f1"], value
    # The selected head is evaluated once on test. Neither preprocessing nor a
    # classifier is refitted using validation/test text.
    margins = _scores(best, features["test"], len(labels))
    predictions = (margins >= 0).astype(int)
    test_metrics = _metrics(truth["test"], predictions, labels)
    dataset_hash = hashlib.sha256(Path(records_file).read_bytes()).hexdigest()
    artifact = {"schema_version": 1, "pipeline_version": FRAMING_PIPELINE_VERSION,
                "input_version": FRAMING_INPUT_VERSION, "purpose": "research",
                "sklearn_version": sklearn.__version__, "labels": labels,
                "features": space, "classifier": best, "threshold": 0.0,
                "selected_C": best_c, "records_sha256": dataset_hash,
                "annotation_contract": "approved_human_spans_v1", "notice": NOTICE}
    report = {"schema_version": 1, "status": "completed", "purpose": "research",
              "experiment_id": experiment_id, "pipeline_version": FRAMING_PIPELINE_VERSION,
              "input_version": FRAMING_INPUT_VERSION, "labels": labels,
              "sklearn_version": sklearn.__version__, "records_sha256": dataset_hash,
              "selected_C": best_c, "selection": "validation_macro_f1_first_tie",
              "candidates": candidate_results, "test": test_metrics,
              "counts": {split: {"n": len(rows), "positive": dict(Counter(
                  label for row in rows for label in row["labels"] if label in labels)),
                  "groups": len({row["group_id"] for row in rows})}
                         for split, rows in splits.items()},
              "uids": {split: [row["uid"] for row in rows] for split, rows in splits.items()},
              "reviewers": sorted({row["review"]["reviewer_name"] for row in records}),
              "feature_count": int(features["train"].shape[1]),
              "threshold": 0.0, "notice": NOTICE,
              "annotation_notice": "Attributed human review declarations and cited context were "
                                   "validated structurally; their truth was not independently checked."}
    (run_dir / "models").mkdir(parents=True)
    (run_dir / "evaluations").mkdir()
    model_path = run_dir / "models/framing_linear_svm.joblib"
    joblib.dump(artifact, model_path, compress=3)
    loaded = load_framing_artifact(model_path)
    reloaded = _scores(loaded["classifier"], loaded["features"].transform(texts["test"]), len(labels))
    report["reload_predictions_match"] = bool(np.array_equal(predictions, (reloaded >= 0).astype(int)))
    if not report["reload_predictions_match"]:
        raise ValueError("saved framing artifact failed prediction reload verification")
    report["artifact_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
    _json(run_dir / "experiment.json", report)
    _json(run_dir / "evaluations/test.json", test_metrics)
    _json(run_dir / "predictions.json", [{"uid": row["uid"], "truth": [
        label for label in labels if label in row["labels"]], "predicted": [
            label for i, label in enumerate(labels) if predictions[j, i]],
        "margins": {label: float(margins[j, i]) for i, label in enumerate(labels)}}
                                         for j, row in enumerate(splits["test"])])
    return run_dir, report


def load_framing_artifact(path):
    """Load a locally generated artifact; never load untrusted joblib/pickle."""
    import joblib
    try:
        artifact = joblib.load(path)
    except (EOFError, pickle.UnpicklingError, struct.error, KeyError, IndexError,
            AttributeError) as exc:
        raise ValueError("Framing model artifact could not be read. Retrain or restore the "
                         "complete locally generated model file.") from exc
    _validate_artifact(artifact)
    return artifact


def _validate_artifact(artifact):
    import sklearn
    from sklearn.multiclass import OneVsRestClassifier

    if (not isinstance(artifact, dict) or artifact.get("schema_version") != 1
            or artifact.get("pipeline_version") != FRAMING_PIPELINE_VERSION
            or artifact.get("input_version") != FRAMING_INPUT_VERSION
            or artifact.get("purpose") != "research"
            or artifact.get("annotation_contract") != "approved_human_spans_v1"
            or artifact.get("threshold") != 0.0):
        raise ValueError("incompatible framing artifact or annotation provenance")
    if artifact.get("sklearn_version") != sklearn.__version__:
        raise ValueError("framing artifact requires the scikit-learn version used to train it")
    if not isinstance(artifact.get("labels"), list):
        raise ValueError("framing artifact labels must be an ordered list")
    labels = _labels(artifact["labels"])
    if (not isinstance(artifact.get("features"), LabFeatureSpace)
            or artifact["features"].arm != "text_only"
            or not isinstance(artifact.get("classifier"), OneVsRestClassifier)
            or not hasattr(artifact["classifier"], "estimators_")
            or len(artifact["classifier"].estimators_) != len(labels)):
        raise ValueError("framing artifact has incompatible features or classifier")


def predict_framing(artifact, text, caption=""):
    """Return independent label decisions and native margins, without risk scores."""
    _validate_artifact(artifact)
    assembled = framing_text(text, caption)
    labels = _labels(artifact.get("labels", []))
    scores = _scores(artifact["classifier"], artifact["features"].transform([assembled]), len(labels))[0]
    return {"labels": [label for i, label in enumerate(labels) if scores[i] >= 0],
            "margins": {label: float(scores[i]) for i, label in enumerate(labels)},
            "threshold": 0.0, "score_kind": "decision_margin", "notice": NOTICE}
