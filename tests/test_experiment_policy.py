"""AI-assisted pilot labels are usable only in an explicit internal smoke."""

import importlib.util
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fipd.curation.splits import save_manifest, split_by_cluster
from fipd.datasets.audit import audit_records
from fipd.models.config import TextConfig
from fipd.schema.records import FactCheckRecord

spec = importlib.util.spec_from_file_location("policy_ablation", ROOT / "scripts/07_ablation.py")
ablation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ablation)


@pytest.fixture
def inputs(tmp_path):
    records = [FactCheckRecord(source="fixture", source_id=str(i), url=f"https://example.test/{i}",
                               label="fake" if i < 3 else "real", claim_text="An investment claim")
               for i in range(6)]
    records[0].raw["ai_assisted_review"] = {"intended_use": "internal_smoke_only"}
    assignment = {r.uid: r.uid for r in records}
    for rec in records:
        rec.raw["cluster_id"] = rec.uid
    source = tmp_path / "records.jsonl"
    source.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")
    splits, report = split_by_cluster(records, assignment, ratios=(1/3, 1/3, 1/3))
    manifest = save_manifest(splits, report, tmp_path, assignment)
    return records, source, manifest


def test_provisional_labels_refuse_normal_research_input(inputs):
    _, source, manifest = inputs
    with pytest.raises(ValueError, match="not research data"):
        ablation.load_experiment_inputs(source, manifest, TextConfig())


def test_explicit_internal_smoke_retains_normal_manifest_checks(inputs):
    _, source, manifest = inputs
    splits, meta = ablation.load_experiment_inputs(source, manifest, TextConfig(),
                                                  allow_internal_smoke=True)
    assert sum(map(len, splits.values())) == 6
    assert meta["leakage_ok"] is True


def test_audit_never_calls_ai_pilot_research_ready(inputs, tmp_path):
    records, _, _ = inputs
    report, _ = audit_records(records, tmp_path)
    assert report["provisional_ai_labelled_records"] == 1
    assert "provisional_ai_labels_not_research_ready" in report["automatic_blockers"]
