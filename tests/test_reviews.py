"""Human review import, conflict handling and persistence through curation."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.curation.labels import assign_label, manual_review_decision, map_verdict  # noqa: E402
from fipd.datasets.reviews import apply_reviews  # noqa: E402
from fipd.schema.records import FactCheckRecord, ImageAsset  # noqa: E402


def candidate(tmp_path):
    Image.new("RGB", (240, 120), (70, 120, 160)).save(tmp_path / "approved.png")
    Image.new("RGB", (180, 100), (130, 170, 190)).save(tmp_path / "other.png")
    return FactCheckRecord(source="example", source_id="1", url="https://example.org/1",
                           title="Fact check: fake official claim", claim_text="Archive description",
                           verdict_raw="False", images=[
                               ImageAsset(url="https://img/1", local_path="approved.png", role="unknown"),
                               ImageAsset(url="https://img/2", local_path="other.png", role="creative")])


def review(rec, **updates):
    row = {"uid": rec.uid, "reviewer_task_relevant": "yes", "reviewer_label": "real",
           "reviewer_image_usable": "yes", "reviewer_image_path": "approved.png",
           "reviewer_name": "Reviewer A", "reviewer_notes": "Verified original creative"}
    row.update(updates)
    return row


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("verdict,expected", [
    ("Not true", "fake"), ("Half true", "fake"), ("Partly False", "fake"),
    ("Partially false", "fake"), ("असत्य", "fake"), ("सत्य", "real"),
    ("True", "real"), ("Mostly true", "real"), ("Unverified", "unknown"),
    ("Inconclusive", "unknown"), ("Insufficient evidence", "unknown"), ("", "unknown"),
])
def test_verdict_ordering(verdict, expected):
    assert map_verdict(verdict) == expected


def test_explicit_import_preserves_source_and_curated_decisions(tmp_path):
    rec = candidate(tmp_path)
    before = rec.to_dict()
    accepted, report = apply_reviews([rec], [review(rec)], tmp_path, "a" * 64, "b" * 64)
    assert len(accepted) == 1 and report["accepted_labels"] == {"real": 1}
    result = accepted[0]
    assert rec.to_dict() == before
    assert result.title == result.claim_text == ""
    assert len(result.images) == 1 and result.images[0].local_path == "approved.png"
    assert result.images[0].role == "creative"
    assert result.raw["manual_review"]["original_title"] == before["title"]
    assert len(result.raw["manual_review"]["original_images"]) == 2
    assert result.url == rec.url and result.uid == rec.uid
    assert assign_label(result) == "real"  # overrides the preserved False source verdict
    assert manual_review_decision(result)["task_relevant"] is True
    result.raw["manual_review"]["reviewer_name"] = ""
    assert manual_review_decision(result) is None
    assert assign_label(result) == "fake"


def test_pending_conflicts_unknown_uid_and_missing_images_are_excluded(tmp_path):
    rec = candidate(tmp_path)
    pending = review(rec, reviewer_name="", reviewer_label="")
    accepted, report = apply_reviews([rec], [pending], tmp_path, "a" * 64, "b" * 64)
    assert not accepted and "reviewer_name_required" in report["decisions"][0]["reasons"]
    for rows, reason in [
        ([review(rec), review(rec, reviewer_label="fake")], "duplicate_review_uid"),
        ([review(rec, uid="unknown")], "unknown_uid"),
        ([review(rec, reviewer_image_path="other-missing.png")], "reviewed_image_file_missing"),
        ([review(rec, reviewer_task_relevant="no")], "reviewer_marked_irrelevant"),
        ([review(rec, reviewer_image_path="../outside.png")], "reviewed_image_outside_data_dir"),
    ]:
        accepted, report = apply_reviews([rec], rows, tmp_path, "a" * 64, "b" * 64)
        assert not accepted
        assert reason in report["decisions"][0]["reasons"]


def test_original_caption_requires_explicit_confirmation_and_source(tmp_path):
    rec = candidate(tmp_path)
    row = review(rec, reviewer_caption="Look at this")
    assert not apply_reviews([rec], [row], tmp_path, "a" * 64, "b" * 64)[0]
    row.update(reviewer_caption_is_original="yes", reviewer_caption_source="Original post screenshot")
    accepted, _ = apply_reviews([rec], [row], tmp_path, "a" * 64, "b" * 64)
    assert accepted[0].claim_text == "Look at this"
    assert accepted[0].raw["manual_review"]["original_claim_text"] == "Archive description"


def test_cli_import_then_curate_retains_reviewed_label_and_relevance(tmp_path, monkeypatch):
    rec = candidate(tmp_path)
    records_file = tmp_path / "input.jsonl"
    records_file.write_text(rec.to_json() + "\n", encoding="utf-8")
    reviews_file = tmp_path / "reviews.csv"
    row = review(rec)
    with reviews_file.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    output = tmp_path / "accepted"
    result = load_script("apply_reviews").main([
        "--records", str(records_file), "--reviews", str(reviews_file),
        "--data-dir", str(tmp_path), "--out-dir", str(output)])
    assert result == 0
    with pytest.raises(SystemExit):
        load_script("apply_reviews").main([
            "--records", str(records_file), "--reviews", str(reviews_file),
            "--data-dir", str(tmp_path), "--out-dir", str(output)])
    monkeypatch.setattr(sys, "argv", ["04_curate.py", "--in-dir", str(output),
                                     "--data-dir", str(tmp_path), "--no-dedup"])
    assert load_script("04_curate").main() == 0
    rows = [json.loads(line) for line in (tmp_path / "interim/finance.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["label"] == "real" and rows[0]["is_finance"]
    assert rows[0]["finance_score"] == 0  # manual task decision beats the keyword filter
    report = json.loads((tmp_path / "interim/curation_report.json").read_text())
    assert report["per_source"]["example"]["manual_reviewed"] == 1


@pytest.mark.parametrize("payload", [b"", b"This is not an image"])
def test_invalid_reviewed_image_is_rejected_without_source_mutation(tmp_path, payload):
    rec = candidate(tmp_path)
    (tmp_path / "approved.png").write_bytes(payload)
    original = rec.to_dict()
    accepted, report = apply_reviews([rec], [review(rec)], tmp_path, "a" * 64, "b" * 64)
    assert accepted == []
    assert "reviewed_image_file_invalid" in report["decisions"][0]["reasons"]
    assert rec.to_dict() == original


def test_review_requires_full_pixel_decode_after_container_verification(tmp_path):
    rec = candidate(tmp_path)
    image_path = tmp_path / "approved.png"
    # A JPEG can expose its size and pass verify() despite truncated pixels.
    Image.effect_noise((200, 200), 100).convert("RGB").save(image_path, format="JPEG")
    data = image_path.read_bytes()
    image_path.write_bytes(data[:len(data) // 2])
    accepted, report = apply_reviews([rec], [review(rec)], tmp_path, "a" * 64, "b" * 64)
    assert accepted == []
    assert "reviewed_image_file_invalid" in report["decisions"][0]["reasons"]


def test_hash_mismatch_is_rejected_and_accepted_geometry_and_ocr_notes_are_preserved(tmp_path):
    rec = candidate(tmp_path)
    rec.images[0].sha256 = "0" * 64
    original = rec.to_dict()
    accepted, report = apply_reviews([rec], [review(rec)], tmp_path, "a" * 64, "b" * 64)
    assert not accepted and "reviewed_image_hash_mismatch" in report["decisions"][0]["reasons"]
    assert rec.to_dict() == original
    rec.images[0].sha256 = hashlib.sha256((tmp_path / "approved.png").read_bytes()).hexdigest()
    rec.images[0].width, rec.images[0].height = 1, 1
    accepted, _ = apply_reviews([rec], [review(rec, reviewer_ocr_quality="Hindi line needs correction")],
                                 tmp_path, "a" * 64, "b" * 64)
    assert len(accepted) == 1
    image = accepted[0].images[0]
    assert (image.width, image.height) == (240, 120)
    assert image.sha256 == rec.images[0].sha256
    evidence = accepted[0].raw["manual_review"]
    assert evidence["reviewer_ocr_quality"] == "Hindi line needs correction"
    assert evidence["image_sha256"] == image.sha256
    assert evidence["original_images"][0]["width"] == 1
    assert rec.images[0].width == 1
