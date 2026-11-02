"""Real sklearn fitting/reloading and leakage policies; no model downloads."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
pytest.importorskip("sklearn")

from fipd.models.classical import (
    LabFeatureSpace, MobileNetFeatures, build_classifier, classifier_scores,
    load_lab_artifact, predict_lab,
)
from fipd.schema.records import FactCheckRecord, ImageAsset
from fipd.training.lab import run_lab_experiment


@pytest.fixture
def lab_data(tmp_path):
    records, assignments = [], {}
    for split, count in (("train", 8), ("val", 4), ("test", 4)):
        for i in range(count):
            label = "fake" if i % 2 else "real"
            filename = f"{split}-{i}.png"
            Image.new("RGB", (24 + i, 24), (len(records) * 13, i * 20, 100)).save(tmp_path / filename)
            text = ("guaranteed profit money double telegram" if label == "fake" else
                    "mutual fund market risks long term investment")
            if split == "test":
                text += " heldoutonlytoken"
            record = FactCheckRecord(
                source="fixture", source_id=f"{split}-{i}", url=f"https://example.test/{filename}",
                label=label, title="VERDICT MUST NEVER ENTER FEATURES",
                images=[ImageAsset(url="", local_path=filename, role="creative", ocr_text=text)],
                raw={"manual_review": {"task_relevant": True, "image_usable": True,
                                       "label": label, "reviewer_name": "test fixture"}})
            record.raw["cluster_id"] = record.uid
            records.append(record)
            assignments[record.uid] = split
    records_path = tmp_path / "records.jsonl"
    records_path.write_text("".join(rec.to_json() + "\n" for rec in records), encoding="utf-8")
    manifest = {"splits": assignments, "clusters": {rec.uid: rec.uid for rec in records},
                "n_records": len(records), "n_clusters": len(records), "leakage_ok": True,
                "counts": dict(Counter(assignments.values())),
                "cluster_counts": dict(Counter(assignments.values())),
                "assignment_hash": hashlib.sha256(json.dumps(
                    sorted(assignments.items()), sort_keys=True).encode()).hexdigest()[:16]}
    manifest_path = tmp_path / "split_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    config = {"arms": ["text_only"], "models": ["linear_svm", "logistic_regression", "naive_bayes"],
              "regularization": [0.1, 1], "nb_alpha": [0.1, 1],
              "max_features": 1000, "benchmark_runs": 2,
              "text": {"fields": ["ocr_text", "claim_text"], "transliterate": False}}
    return tmp_path, records_path, manifest_path, config


def test_text_preprocessing_is_fit_only_on_training_and_preserves_hindi():
    texts = ["निवेश बाजार जोखिम", "गारंटी मुनाफा दोगुना पैसा"]
    space = LabFeatureSpace().fit(texts)
    space.transform(["heldoutonlytoken"])
    word = space.vectorizer.transformer_list[0][1]
    character = space.vectorizer.transformer_list[1][1]
    assert "heldoutonlytoken" not in word.vocabulary_
    assert any("ि" in token for token in character.vocabulary_)
    assert space.transform(texts).nnz > 0


def test_image_scaler_never_fits_on_validation_or_test():
    space = LabFeatureSpace("fused").fit(["market risks", "double money"],
                                        np.array([[0, 2], [2, 4]], dtype=np.float32))
    space.transform(["validation"], np.array([[1000, 2000]], dtype=np.float32))
    np.testing.assert_allclose(space.image_scaler.mean_, [1, 3])
    assert space.transform(["validation"], np.array([[3, 5]])).shape[0] == 1
    with pytest.raises(ValueError, match="finite image features"):
        space.transform(["validation"], np.array([[np.nan, 1]]))


def test_svm_uses_signed_margins_and_other_models_use_probability():
    texts = ["mutual fund risk", "investment risk", "double profit", "guaranteed profit"]
    space = LabFeatureSpace().fit(texts)
    features = space.transform(texts)
    for name in ("linear_svm", "logistic_regression", "naive_bayes"):
        classifier = build_classifier(name).fit(features, [0, 0, 1, 1])
        scores, threshold, kind = classifier_scores(classifier, features)
        if name == "linear_svm":
            assert threshold == 0 and kind == "decision_margin" and min(scores) < 0
        else:
            assert threshold == .5 and np.all((scores >= 0) & (scores <= 1))


def test_real_text_training_selection_artifacts_and_prediction_roundtrip(lab_data):
    root, records, manifest, config = lab_data
    run, identity = run_lab_experiment(records, manifest, root, root / "out", config,
                                        experiment_id="text")
    assert identity["status"] == "complete" and len(identity["results"]) == 3
    assert (run / "comparison.csv").is_file()
    for row in identity["results"]:
        path = Path(row["artifact"])
        artifact = load_lab_artifact(path)
        word = artifact["features"].vectorizer.transformer_list[0][1]
        assert "heldoutonlytoken" not in word.vocabulary_
        assert "verdict" not in word.vocabulary_
        result = predict_lab(artifact, "guaranteed double profit telegram")
        assert result["label"] == "fake" and not result["unvalidated"]
        detail = json.loads((run / "evaluations" / f"text_only_{row['model']}.json").read_text())
        assert detail["reload_verified"] is True
        assert len(detail["predictions"]) == 4
        assert row["validation_macro_f1"] == max(
            item["metrics"]["macro_f1"] for item in detail["validation_candidates"])
    with pytest.raises(ValueError, match="already exists"):
        run_lab_experiment(records, manifest, root, root / "out", config, experiment_id="text")


def test_lab_research_refuses_unreviewed_and_provisional_records(lab_data):
    root, records_path, manifest, config = lab_data
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    rows[0]["raw"].pop("manual_review")
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="attributed manual review"):
        run_lab_experiment(records_path, manifest, root, root / "out", config)
    rows[0]["raw"]["ai_assisted_review"] = {"intended_use": "internal_smoke_only"}
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="not research data"):
        run_lab_experiment(records_path, manifest, root, root / "out", config)
    run, _ = run_lab_experiment(records_path, manifest, root, root / "out", config,
                                 internal_pilot=True, models=["linear_svm"])
    artifact_path = next((run / "models").glob("*.joblib"))
    with pytest.raises(ValueError, match="explicit --internal-pilot"):
        load_lab_artifact(artifact_path)
    artifact = load_lab_artifact(artifact_path, allow_internal=True)
    assert predict_lab(artifact, "investment market risk")["unvalidated"] is True


def test_single_class_and_split_image_leakage_cannot_produce_metrics(lab_data):
    root, records_path, manifest, config = lab_data
    rows = [json.loads(line) for line in records_path.read_text().splitlines()]
    for row in rows:
        row["label"] = row["raw"]["manual_review"]["label"] = "fake"
    records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="both real"):
        run_lab_experiment(records_path, manifest, root, root / "out", config)
    assert not (root / "out").exists()


def test_multimodal_heads_share_cohort_and_fused_has_both_feature_types(lab_data, monkeypatch):
    root, records, manifest, config = lab_data

    class FakeEncoder:
        weights_sha256 = "test-only"

        def __init__(self, *args):
            pass

        def encode_images(self, images):
            return np.array([[image.width, image.height] for image in images], dtype=np.float32)

        def encode_paths(self, paths):
            output = []
            for path in paths:
                with Image.open(path) as image:
                    output.append([image.width, image.height])
            return np.array(output, dtype=np.float32)

    import fipd.training.lab as lab
    monkeypatch.setattr(lab, "MobileNetFeatures", FakeEncoder)
    run, identity = run_lab_experiment(records, manifest, root, root / "out", config,
                                        arms=["text_only", "image_only", "fused"],
                                        models=["linear_svm"])
    rows = identity["results"]
    assert len(rows) == 3 and {row["test_n"] for row in rows} == {4}
    assert rows[2]["feature_count"] == rows[0]["feature_count"] + rows[1]["feature_count"]
    fused = load_lab_artifact(Path(rows[2]["artifact"]))
    assert predict_lab(fused, "investment risk", image=Image.new("RGB", (25, 24)),
                       encoder=FakeEncoder())["arm"] == "fused"
    assert len(list((run / "evaluations").glob("*.json"))) == 3


def test_image_weights_are_required_without_attempting_network(tmp_path):
    pytest.importorskip("torchvision")
    with pytest.raises(FileNotFoundError, match="prepare-image-model"):
        MobileNetFeatures(tmp_path / "missing.pth")


def test_text_config_excludes_archive_titles(lab_data):
    root, records, manifest, config = lab_data
    config["text"]["fields"] = ["title"]
    with pytest.raises(ValueError, match="original caption"):
        run_lab_experiment(records, manifest, root, root / "out", config)
