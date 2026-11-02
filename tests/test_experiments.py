"""Offline experiment regressions with real tiny torch training, no downloads."""

from __future__ import annotations

import importlib.util
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from fipd.datasets.examples import Example  # noqa: E402
from fipd.models.config import load_model_config  # noqa: E402
from fipd.schema.records import FactCheckRecord, ImageAsset  # noqa: E402

SPEC = importlib.util.spec_from_file_location("fipd_ablation_script", ROOT / "scripts/07_ablation.py")
ablation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ablation)


def fixture_uid(name):
    return FactCheckRecord(source="fixture", source_id=name, url="").uid


def write_fixture_inputs(root, records, assignments):
    """Real schema rows and independent manifest metadata exercise preflight."""
    records_file = root / "interim/finance.jsonl"
    records_file.write_text("\n".join(rec.to_json() for rec in records) + "\n", encoding="utf-8")
    clusters = {rec.uid: rec.raw.get("cluster_id") or rec.uid for rec in records}
    manifest = {"splits": assignments, "clusters": clusters,
                "assignment_hash": hashlib.sha256(json.dumps(
                    sorted(assignments.items()), sort_keys=True).encode()).hexdigest()[:16],
                "n_records": len(records), "n_clusters": len(set(clusters.values())),
                "counts": dict(Counter(assignments.values())),
                "cluster_counts": {split: len({clusters[uid] for uid, value in assignments.items()
                                                if value == split})
                                   for split in set(assignments.values())},
                "leakage_ok": True}
    (root / "processed/split_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


@pytest.fixture
def cohort_fixture(tmp_path):
    for split, color in (("train", "white"), ("val", "black"), ("test", "red")):
        Image.new("RGB", (8, 8), color).save(tmp_path / f"{split}.png")
    (tmp_path / "bad.png").write_text("not an image", encoding="utf-8")
    splits = {
        split: [Example(f"{split}-{i}", f"claim {i}", i % 2,
                        image_path=f"{split}.png", split=split)
                for i in range(4)]
        for split in ("train", "val", "test")
    }
    splits["train"] += [Example("text-extra", "caption only", 0),
                        Example("corrupt-extra", "bad image", 1, image_path="bad.png")]
    return tmp_path, splits


def test_common_cohort_filters_once_and_records_image_content(cohort_fixture):
    root, splits = cohort_fixture
    cohort, excluded, digest = ablation.common_cohort(splits, root)
    assert [ex.uid for ex in cohort["train"]] == [f"train-{i}" for i in range(4)]
    assert excluded["train"] == ["text-extra", "corrupt-extra"]
    Image.new("RGB", (8, 8), "blue").save(root / "train.png")
    assert ablation.common_cohort(splits, root)[2] != digest


def test_chosen_image_bytes_shared_across_splits_are_rejected(cohort_fixture):
    root, splits = cohort_fixture
    (root / "val.png").write_bytes((root / "train.png").read_bytes())
    with pytest.raises(ValueError, match="chosen image bytes are shared across splits"):
        ablation.common_cohort(splits, root)


@pytest.mark.parametrize("split", ["train", "val", "test"])
@pytest.mark.parametrize("empty", [False, True])
def test_filtered_splits_must_be_nonempty_and_two_class(cohort_fixture, split, empty):
    root, splits = cohort_fixture
    splits[split] = [] if empty else [ex for ex in splits[split] if ex.label == 1]
    with pytest.raises(ValueError, match=rf"{split} must contain both"):
        ablation.common_cohort(splits, root)


@pytest.mark.parametrize("manifest", [
    {"reviewed": False, "uids": ["test-0", "test-1"]},
    {"reviewed": True, "uids": []},
    {"reviewed": True, "uids": ["test-0", "test-0"]},
    {"reviewed": True, "uids": ["train-0", "test-1"]},
    {"reviewed": True, "uids": ["test-1", "test-3"]},
])
def test_invalid_reviewed_subsets_cannot_be_reported(cohort_fixture, manifest):
    root, splits = cohort_fixture
    path = root / "subset.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError):
        ablation.reviewed_subset(path, splits["test"])


@pytest.fixture
def offline_experiment(monkeypatch, cohort_fixture):
    torch = pytest.importorskip("torch")
    from fipd.datasets import torch_data
    from fipd.models import arms

    root, splits = cohort_fixture
    (root / "interim").mkdir()
    (root / "processed").mkdir()
    records, assignments = [], {}
    for split, examples in splits.items():
        for ex in examples:
            rec = FactCheckRecord(source="fixture", source_id=ex.uid,
                                  url=f"https://fixture.test/{ex.uid}", claim_text=ex.text,
                                  label="fake" if ex.label else "real")
            if ex.image_path:
                rec.images = [ImageAsset(url="https://fixture.test/image.png",
                                         local_path=ex.image_path, role="creative")]
            rec.raw["cluster_id"] = rec.uid
            records.append(rec)
            assignments[rec.uid] = split
    write_fixture_inputs(root, records, assignments)
    cfg = load_model_config()
    cfg.training.epochs = 1
    cfg.training.batch_size = 2
    cfg.training.seeds = [42]
    cfg.training.num_workers = 0
    cfg.training.lr = cfg.training.head_lr = 0.01
    monkeypatch.setattr(ablation, "load_model_config", lambda path: cfg)
    initial_states, dataset_ids = [], []

    class TinyDataset:
        def __init__(self, examples, needs_image):
            self.examples, self.needs_image = examples, needs_image

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, index):
            ex = self.examples[index]
            features = torch.tensor([1 - ex.label, ex.label], dtype=torch.float32)
            out = {"uid": ex.uid, "label": torch.tensor(ex.label),
                   "input_ids": features, "attention_mask": torch.ones(2)}
            if self.needs_image:
                out["pixel_values"] = features
            return out

    def dataset(examples, **kwargs):
        dataset_ids.append([ex.uid for ex in examples])
        return TinyDataset(examples, kwargs["needs_image"])

    class TinyModel(torch.nn.Module):
        def __init__(self, arm):
            super().__init__()
            self.arm = arm
            self.classifier = torch.nn.Linear(2, 2)

        def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
            features = pixel_values if self.arm == "image_only" else input_ids
            return self.classifier(features.float())

    def build(arm, arch, **kwargs):
        model = TinyModel(arm)
        initial_states.append({key: value.clone() for key, value in model.state_dict().items()})
        return model

    monkeypatch.setattr(torch_data, "build_dataset", dataset)
    monkeypatch.setattr(arms, "build_arm", build)
    return root, records, cfg, initial_states, dataset_ids


def report_dir(root, variant, experiment):
    return root / "processed/ablation" / variant / experiment


def test_all_arms_share_cohort_seed_and_full_subset_predictions(offline_experiment):
    root, _, _, states, datasets = offline_experiment
    subset = root / "subset.json"
    subset.write_text(json.dumps({"name": "gate4", "reviewed": True,
                                  "uids": [fixture_uid("test-1"), fixture_uid("test-0")]}),
                      encoding="utf-8")
    assert ablation.main(["--data-dir", str(root), "--experiment-id", "first",
                          "--reviewed-subset", str(subset)]) == 0
    import torch

    # Each arm's random head starts from the declared seed, before construction.
    assert all(torch.equal(states[0][key], state[key]) for state in states for key in state)
    assert datasets == [[fixture_uid(f"{split}-{i}") for i in range(4)]
                        for _ in range(3) for split in ("train", "val", "test")]
    reports = report_dir(root, "student", "first")
    identity = json.loads((reports / "experiment.json").read_text())["identity"]
    assert identity["variant"] == "student"
    assert identity["cohort_ids"]["train"] == [fixture_uid(f"train-{i}") for i in range(4)]
    assert identity["config"]["input_pipeline_version"] == ablation.INPUT_PIPELINE_VERSION
    for arm in ("text_only", "image_only", "fused"):
        full = json.loads((reports / f"evaluations/full/{arm}_seed42.json").read_text())
        subset_file = next(reports.glob(f"evaluations/reviewed-*/{arm}_seed42.json"))
        reviewed = json.loads(subset_file.read_text())
        assert reviewed["predictions"] == full["predictions"][:2]
        assert reviewed["metrics"]["threshold"] == full["metrics"]["threshold"]
        checkpoint = torch.load(root / f"checkpoints/student/first/{arm}_seed42.pt",
                                weights_only=True)
        assert checkpoint["cohort_ids"] == identity["cohort_ids"]
        assert checkpoint["config"]["architecture"] == identity["config"]["architecture"]
        assert full["metrics"]["n"] == 4
    # An explicit duplicate experiment fails before constructing/training anything.
    assert ablation.main(["--data-dir", str(root), "--experiment-id", "first"]) == 1
    assert len(states) == 3


def test_repeated_seed_is_reproducible_and_variants_are_isolated(offline_experiment):
    root, _, _, states, _ = offline_experiment
    for variant, experiment in (("student", "one"), ("student", "two"), ("teacher", "one")):
        assert ablation.main(["--data-dir", str(root), "--experiment-id", experiment,
                              "--variant", variant, "--arms", "fused"]) == 0
    import torch

    first = torch.load(root / "checkpoints/student/one/fused_seed42.pt", weights_only=True)
    second = torch.load(root / "checkpoints/student/two/fused_seed42.pt", weights_only=True)
    assert all(torch.equal(first["state_dict"][key], second["state_dict"][key])
               for key in first["state_dict"])
    assert len(states) == 3
    assert report_dir(root, "teacher", "one").is_dir()
    assert not (root / "checkpoints/fused_seed42.pt").exists()


def test_evaluation_only_reuses_checkpoint_and_preserves_history(offline_experiment, monkeypatch):
    root, records, _, _, _ = offline_experiment
    from fipd.training import loop

    args = ["--data-dir", str(root), "--experiment-id", "saved", "--arms", "fused"]
    assert ablation.main(args) == 0
    checkpoint = root / "checkpoints/student/saved/fused_seed42.pt"
    checkpoint_bytes = checkpoint.read_bytes()
    run_path = report_dir(root, "student", "saved") / "runs/fused_seed42.json"
    before = json.loads(run_path.read_text())
    subset = root / "subset.json"
    subset.write_text(json.dumps({"reviewed": True,
                                  "uids": [fixture_uid("test-0"), fixture_uid("test-1")]}),
                      encoding="utf-8")

    def forbid_training(*args, **kwargs):
        pytest.fail("evaluation-only must never train")

    monkeypatch.setattr(loop, "train_arm", forbid_training)
    assert ablation.main(args + ["--evaluate-only", "--reviewed-subset", str(subset)]) == 0
    after = json.loads(run_path.read_text())
    assert checkpoint.read_bytes() == checkpoint_bytes
    assert after["history"] == before["history"]
    assert after["decision_threshold"] == before["decision_threshold"]
    assert len(after["evaluations"]) == 2
    # Provenance verification detects changed corpus content before model work.
    records[0].claim_text = "changed claim in a valid schema row"
    (root / "interim/finance.jsonl").write_text(
        "\n".join(rec.to_json() for rec in records) + "\n", encoding="utf-8")
    assert ablation.main(args + ["--evaluate-only"]) == 1


def test_empty_test_and_unreviewed_gate4_fail_without_artifacts(offline_experiment):
    root, records, _, states, _ = offline_experiment
    args = ["--data-dir", str(root), "--experiment-id", "invalid"]
    assert ablation.main(args + ["--image-only-claims"]) == 1
    for rec in records:
        if rec.source_id.startswith("test-"):
            rec.images = []
    (root / "interim/finance.jsonl").write_text(
        "\n".join(rec.to_json() for rec in records) + "\n", encoding="utf-8")
    assert ablation.main(args) == 1
    assert not states
    assert not report_dir(root, "student", "invalid").exists()


@pytest.mark.parametrize("case, message", [
    ("missing_uid", "UID coverage"),
    ("extra_uid", "UID coverage"),
    ("missing_cluster", "UID coverage"),
    ("invalid_split", "invalid split name"),
    ("stale_merged_cluster", "current claim cluster.*spans splits"),
    ("stale_same_split_cluster", "cluster assignments differ"),
    ("tampered_hash", "assignment hash"),
    ("tampered_record_count", "counts do not match"),
    ("tampered_cluster_count", "counts do not match"),
    ("tampered_split_counts", "counts do not match"),
    ("failed_leakage_flag", "failed or missing leakage check"),
])
def test_manifest_preflight_rechecks_current_corpus(offline_experiment, case, message):
    root, records, cfg, states, _ = offline_experiment
    manifest_file = root / "processed/split_manifest.json"
    manifest = json.loads(manifest_file.read_text())
    train_uid, test_uid = fixture_uid("train-0"), fixture_uid("test-0")
    if case == "missing_uid":
        del manifest["splits"][train_uid]
    elif case == "extra_uid":
        manifest["splits"]["absent-record"] = "train"
    elif case == "missing_cluster":
        del manifest["clusters"][train_uid]
    elif case == "invalid_split":
        manifest["splits"][train_uid] = "holdout"
    elif case in {"stale_merged_cluster", "stale_same_split_cluster"}:
        target = test_uid if case == "stale_merged_cluster" else fixture_uid("train-1")
        next(rec for rec in records if rec.uid == train_uid).raw["cluster_id"] = target
        (root / "interim/finance.jsonl").write_text(
            "\n".join(rec.to_json() for rec in records) + "\n", encoding="utf-8")
    elif case == "tampered_hash":
        manifest["assignment_hash"] = "tampered"
    elif case == "tampered_record_count":
        manifest["n_records"] += 1
    elif case == "tampered_cluster_count":
        manifest["n_clusters"] += 1
    elif case == "tampered_split_counts":
        manifest["counts"]["train"] += 1
    elif case == "failed_leakage_flag":
        manifest["leakage_ok"] = False
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        ablation.load_experiment_inputs(root / "interim/finance.jsonl", manifest_file, cfg.text)
    assert ablation.main(["--data-dir", str(root), "--experiment-id", "stale"]) == 1
    assert not states
    assert not report_dir(root, "student", "stale").exists()


@pytest.mark.parametrize("case", ["invalid_json", "wrong_type", "bad_field", "duplicate_uid",
                                 "serialized_uid_mismatch"])
def test_malformed_rows_never_silently_disappear(offline_experiment, case):
    root, records, cfg, states, _ = offline_experiment
    path = root / "interim/finance.jsonl"
    lines = [rec.to_json() for rec in records]
    if case == "invalid_json":
        lines.append("{broken-json")
    elif case == "wrong_type":
        lines.append("[]")
    elif case == "bad_field":
        row = records[0].to_dict()
        row["title"] = {"invalid": "text"}
        lines[0] = json.dumps(row)
    elif case == "duplicate_uid":
        lines.append(records[0].to_json())
    else:
        row = records[0].to_dict()
        row["uid"] = "wrong-serialized-uid"
        lines[0] = json.dumps(row)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="finance.jsonl:[0-9]+: invalid record"):
        ablation.load_experiment_inputs(path, root / "processed/split_manifest.json", cfg.text)
    assert ablation.main(["--data-dir", str(root), "--experiment-id", "malformed"]) == 1
    assert not states
    assert not report_dir(root, "student", "malformed").exists()


def test_train_loop_rejects_empty_loaders_and_existing_checkpoint(tmp_path):
    torch = pytest.importorskip("torch")
    from fipd.training.loop import train_arm

    model = torch.nn.Linear(2, 2)
    with pytest.raises(ValueError, match="non-empty"):
        train_arm(model, [], [], arm="text_only", seed=42)
    checkpoint = tmp_path / "text_only_seed42.pt"
    checkpoint.write_bytes(b"existing selected model")
    with pytest.raises(FileExistsError):
        train_arm(model, [1], [1], arm="text_only", seed=42, checkpoint_dir=tmp_path)
    assert checkpoint.read_bytes() == b"existing selected model"
