"""Provisional pilot artifacts must not masquerade as human-reviewed data."""

import hashlib
import importlib.util
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fipd.schema.records import FactCheckRecord, ImageAsset

spec = importlib.util.spec_from_file_location("assisted_pilot", ROOT / "scripts/build_assisted_pilot.py")
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


@pytest.fixture
def pilot(tmp_path):
    source = tmp_path / "source.jsonl"
    records, choices = [], []
    for index in range(6):
        image_path = f"image{index}.png"
        Image.new("RGB", (10, 10), (index * 30, 100, 150)).save(tmp_path / image_path)
        rec = FactCheckRecord(source="fixture", source_id=str(index), url=f"https://example.test/{index}",
                              title="Factchecker verdict title", claim_text="Archive summary",
                              verdict_raw="False", images=[ImageAsset(url="https://img.test/a",
                              local_path=image_path, role="unknown", ocr_text="old OCR")])
        records.append(rec)
        choices.append({"uid": rec.uid, "source_records": str(source),
                        "selected_image_path": image_path,
                        "image_sha256": hashlib.sha256((tmp_path / image_path).read_bytes()).hexdigest(),
                        "suggested_label": "fake" if index < 3 else "real",
                        "cluster_id": rec.uid, "scope_evidence": "Visible investment promotion",
                        "source_label_evidence": "Fixture evidence only"})
    source.write_text("".join(rec.to_json() + "\n" for rec in records), encoding="utf-8")
    return tmp_path, source, {"reviewer_kind": "ai_assisted",
                              "intended_use": "internal_smoke_only", "records": choices}


def test_pilot_keeps_sources_and_excludes_verdict_text(pilot):
    data, source, manifest = pilot
    before = source.read_bytes()
    records, report = builder.build_pilot(manifest, data)
    assert len(records) == 6 and report["human_approved"] is False
    assert source.read_bytes() == before
    for rec in records:
        assert rec.title == rec.claim_text == rec.verdict_raw == ""
        assert "manual_review" not in rec.raw
        assert rec.raw["ai_assisted_review"]["human_approved"] is False
        assert rec.raw["ai_assisted_review"]["original_title"]
        assert rec.images[0].ocr_text is None


def test_research_intent_cannot_use_provisional_builder(pilot):
    data, _, manifest = pilot
    manifest["intended_use"] = "final_research_accuracy"
    with pytest.raises(ValueError, match="internal-smoke"):
        builder.build_pilot(manifest, data)


def test_item_metadata_cannot_forge_human_approval(pilot):
    data, _, manifest = pilot
    manifest["records"][0].update(human_approved=True, reviewer_kind="human",
                                    intended_use="final_research_accuracy")
    records, _ = builder.build_pilot(manifest, data)
    decision = next(r.raw["ai_assisted_review"] for r in records
                    if r.uid == manifest["records"][0]["uid"])
    assert decision["human_approved"] is False
    assert decision["reviewer_kind"] == "ai_assisted"
    assert decision["intended_use"] == "internal_smoke_only"


def test_pilot_rejects_image_changed_since_visual_review(pilot):
    data, _, manifest = pilot
    Image.new("RGB", (10, 10), "red").save(data / manifest["records"][0]["selected_image_path"])
    with pytest.raises(ValueError, match="byte hash"):
        builder.build_pilot(manifest, data)


def test_pilot_requires_three_proposed_groups_per_class(pilot):
    data, _, manifest = pilot
    for item in manifest["records"][:3]:
        item["cluster_id"] = "same-claim"
    with pytest.raises(ValueError, match="three proposed groups"):
        builder.build_pilot(manifest, data)
