"""Synthetic fixtures test engineering behavior, never detection performance."""
from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("sklearn")

from fipd.training.framing import (
    load_framing_artifact, load_framing_records, predict_framing, run_framing_experiment,
)

LABELS = ["return_promise", "urgency"]


def annotation(uid, split, text, labels):
    evidence = []
    for label in labels:
        quote = "guaranteed profit" if label == "return_promise" else "buy today"
        start = text.index(quote)
        evidence.append({"label": label, "field": "text", "start": start,
                         "end": start + len(quote), "quote": quote,
                         "rationale": "Synthetic annotation fixture for engineering tests."})
    return {"schema_version": 1, "uid": uid, "group_id": uid, "split": split,
            "text": text, "caption": "", "labels": labels, "evidence": evidence,
            "as_of": "2026-10-20", "review": {"method": "human", "status": "approved",
                "reviewer_name": "Synthetic fixture", "reviewed_at": "2026-10-21"}}


@pytest.fixture
def records():
    rows = []
    for split in ("train", "val", "test"):
        for i, (wording, labels) in enumerate([
            ("guaranteed profit from this company", ["return_promise"]),
            ("buy today before the offer ends", ["urgency"]),
            ("guaranteed profit buy today", LABELS),
            ("revenue improved but market losses remain possible", []),
            ("long term investment could outperform with risks", []),
            ("guaranteed profit the outcome is certain", ["return_promise"]),
        ]):
            text = f"{wording}. {split} creative {i}"
            if split == "test":
                text += " testvocabularyonly"
            rows.append(annotation(f"{split}-{i}", split, text, list(labels)))
    return rows


def write_rows(tmp_path, rows, name="annotations.jsonl"):
    path = tmp_path / name
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def test_reviews_spans_and_negative_labels_are_required(tmp_path, records):
    path = write_rows(tmp_path, records)
    assert len(load_framing_records(path, LABELS)) == 18
    damaged = copy.deepcopy(records)
    damaged[3]["review"]["method"] = "rule"
    with pytest.raises(ValueError, match="approved human review"):
        load_framing_records(write_rows(tmp_path, damaged), LABELS)
    damaged = copy.deepcopy(records)
    damaged[0]["evidence"][0]["quote"] = "incorrect quote"
    with pytest.raises(ValueError, match="exactly match"):
        load_framing_records(write_rows(tmp_path, damaged), LABELS)
    damaged = copy.deepcopy(records)
    damaged[0]["evidence"] = []
    with pytest.raises(ValueError, match="every positive label"):
        load_framing_records(write_rows(tmp_path, damaged), LABELS)


@pytest.mark.parametrize("mutation,expected", [
    (lambda rows: rows[6].update(group_id=rows[0]["group_id"]), "group_id spans"),
    (lambda rows: rows[9].update(text=rows[3]["text"].upper(), caption="different caption"),
     "exact promotion text"),
    (lambda rows: rows[6].update(uid=rows[0]["uid"]), "duplicate uid"),
    (lambda rows: rows[0].update(labels=["fake"]), "known unique"),
    (lambda rows: rows[0].update(labels=[{}]), "known unique"),
    (lambda rows: rows[0].update(split=[]), "line 1"),
    (lambda rows: rows[0]["evidence"][0].update(start=True), "character start"),
])
def test_split_leakage_and_bad_annotations_fail(tmp_path, records, mutation, expected):
    mutation(records)
    with pytest.raises(ValueError, match=expected):
        load_framing_records(write_rows(tmp_path, records), LABELS)


def test_selective_context_requires_material_evidence_available_at_publication(tmp_path, records):
    for row in records:
        if row["labels"]:
            row["labels"].append("selective_context")
            row["evidence"].append({"label": "selective_context", "field": "text",
                "start": 0, "end": 5, "quote": row["text"][:5],
                "rationale": "Reviewer compared periods; test fixture only.",
                "source_url": "https://example.test/filing", "source_title": "Test filing",
                "published_at": "2026-10-19", "material_context": "Context changes interpretation."})
    path = write_rows(tmp_path, records)
    assert len(load_framing_records(path, ["selective_context"])) == 18
    records[0]["evidence"][-1]["published_at"] = "2026-10-22"
    with pytest.raises(ValueError, match="published after"):
        load_framing_records(write_rows(tmp_path, records), ["selective_context"])
    records[0]["evidence"][-1]["published_at"] = "2026-10-19"
    records[0]["evidence"][-1].pop("material_context")
    with pytest.raises(ValueError, match="material_context"):
        load_framing_records(write_rows(tmp_path, records), ["selective_context"])


def test_every_requested_label_has_both_classes_in_each_split(tmp_path, records):
    with pytest.raises(ValueError, match="positive and negative.*social_proof"):
        load_framing_records(write_rows(tmp_path, records), ["social_proof"])


@pytest.mark.parametrize("url", ["https://name:password@example.test/file",
    "https://example.test /filing", "https://bad..example.test/file", "https://localhost/file",
    "https://127.0.0.1/file", "https://example.test:99999/file", "https://-bad.example.test/file"])
def test_selective_context_rejects_malformed_source_urls(tmp_path, records, url):
    row = records[0]
    row["labels"].append("selective_context")
    row["evidence"].append({"label": "selective_context", "field": "text", "start": 0,
        "end": 5, "quote": row["text"][:5], "rationale": "Reviewed fixture context.",
        "source_url": url, "source_title": "Filing", "published_at": "2026-10-19",
        "material_context": "Relevant context."})
    with pytest.raises(ValueError, match="source_url"):
        load_framing_records(write_rows(tmp_path, records), LABELS)


def test_training_saves_reloadable_multilabel_model_and_train_only_features(tmp_path, records):
    path = write_rows(tmp_path, records)
    run, report = run_framing_experiment(path, tmp_path / "runs", labels=LABELS,
        regularization=[0.1, 1], max_features=1000, experiment_id="fixture")
    assert report["status"] == "completed"
    assert report["reload_predictions_match"] is True
    assert report["selected_C"] == max(report["candidates"],
        key=lambda value: value["validation"]["macro_f1"])["C"]
    assert report["test"]["n"] == 6
    assert set(report["test"]["per_label"]) == set(LABELS)
    assert report["test"]["per_label"]["urgency"]["support"] == 2
    artifact = load_framing_artifact(run / "models/framing_linear_svm.joblib")
    word = artifact["features"].vectorizer.transformer_list[0][1]
    assert "testvocabularyonly" not in word.vocabulary_
    output = predict_framing(artifact, "guaranteed profit buy today")
    assert set(output["margins"]) == set(LABELS)
    assert output["threshold"] == 0
    assert output["score_kind"] == "decision_margin"
    assert "not probabilities" in output["notice"]
    assert "risk_score" not in output
    assert all(np.isfinite(value) for value in output["margins"].values())
    assert (run / "evaluations/test.json").is_file()
    assert len(json.loads((run / "predictions.json").read_text())) == 6
    with pytest.raises(ValueError, match="already exists"):
        run_framing_experiment(path, tmp_path / "runs", labels=LABELS, experiment_id="fixture")
    artifact["pipeline_version"] = "old-version"
    with pytest.raises(ValueError, match="incompatible"):
        predict_framing(artifact, "market risks")


def test_single_label_training_has_positive_class_metrics(tmp_path, records):
    path = write_rows(tmp_path, records)
    run, report = run_framing_experiment(path, tmp_path / "runs", labels=["urgency"],
                                        regularization=[1], experiment_id="single-fixture")
    assert report["test"]["per_label"]["urgency"]["support"] == 2
    artifact = load_framing_artifact(run / "models/framing_linear_svm.joblib")
    assert list(predict_framing(artifact, "buy today")["margins"]) == ["urgency"]


def test_cli_missing_annotations_is_actionable(tmp_path, capsys):
    script = Path(__file__).resolve().parents[1] / "scripts/12_train_framing.py"
    spec = importlib.util.spec_from_file_location("train_framing_cli", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--records", str(tmp_path / "missing.jsonl")]) == 1
    assert "docs/FRAMING_LABELS.md" in capsys.readouterr().err
    with pytest.raises(ValueError, match="finite positive"):
        run_framing_experiment("missing", regularization=[float("nan")])


@pytest.mark.parametrize("content", [b"", b"\x80\x04\x95", b"not a model"])
def test_corrupt_artifact_load_is_a_clean_validation_error(tmp_path, content):
    path = tmp_path / "broken.joblib"
    path.write_bytes(content)
    with pytest.raises(ValueError, match="artifact could not be read"):
        load_framing_artifact(path)


def test_truncated_compressed_artifact_and_audit_cli_fail_cleanly(tmp_path, capsys):
    import joblib

    path = tmp_path / "truncated.joblib"
    joblib.dump({"test_fixture": True}, path, compress=3)
    content = path.read_bytes()
    path.write_bytes(content[:len(content) // 2])
    with pytest.raises(ValueError, match="complete locally generated"):
        load_framing_artifact(path)
    script = Path(__file__).resolve().parents[1] / "scripts/13_audit_promotion.py"
    spec = importlib.util.spec_from_file_location("audit_cli_corrupt_framing_fixture", script)
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    assert cli.main(["--text", "This stock will double.", "--framing-artifact", str(path)]) == 1
    stderr = capsys.readouterr().err
    assert "artifact could not be read" in stderr
    assert "Traceback" not in stderr
