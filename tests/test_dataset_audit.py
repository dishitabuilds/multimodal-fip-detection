"""Offline regressions for corpus preservation, provenance and review exports."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.datasets.audit import (  # noqa: E402
    audit_records, inspect_image_dimensions, read_inputs, select_inputs, write_review,
)
from fipd.schema.records import FactCheckRecord, ImageAsset  # noqa: E402


def record(source_id="1", **kwargs):
    return FactCheckRecord(source="example", source_id=source_id,
                           url=f"https://example.org/{source_id}", **kwargs)


def save(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")
    return path


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_enrichment_precedence_is_complete_and_deterministic(tmp_path):
    raw_record = record(title="discovery")
    enriched_record = record(title="full article", images=[ImageAsset(url="https://img/1",
                                                                      local_path="images/1.png")])
    raw = save(tmp_path / "raw.jsonl", [record("2"), raw_record])
    enriched = save(tmp_path / "enriched.jsonl", [enriched_record])
    selected, report = select_inputs([raw], [enriched])
    assert report["safe_to_write"]
    assert [r.uid for r in selected] == sorted(r.uid for r in selected)
    winner = next(r for r in selected if r.uid == raw_record.uid)
    assert winner.to_dict() == enriched_record.to_dict()
    assert report["selections"][winner.uid]["selected"]["tier"] == "enriched"
    assert len(report["duplicate_uids"]) == 1
    assert select_inputs([raw, raw], [enriched])[1] == report


@pytest.mark.parametrize("override,reason", [
    ({"label": "real"}, "conflicting_labels_or_verdicts"),
    ({"verdict_raw": "Genuine"}, "conflicting_labels_or_verdicts"),
    ({"url": "https://different.org/article"}, "conflicting_provenance_urls"),
    ({"label": "unknown"}, "enrichment_would_discard_known_label"),
])
def test_conflicts_are_reported_before_selection(tmp_path, override, reason):
    original = record(label="fake")
    changed = record(label="fake")
    for key, value in override.items():
        setattr(changed, key, value)
    selected, report = select_inputs([save(tmp_path / "raw.jsonl", [original])],
                                      [save(tmp_path / "enriched.jsonl", [changed])])
    assert not selected
    assert not report["safe_to_write"]
    assert reason in report["conflicts"][0]["reasons"]
    assert len(report["conflicts"][0]["variants"]) == 2


def test_same_tier_ambiguity_and_invalid_uid_are_not_silently_dropped(tmp_path):
    raw = save(tmp_path / "raw.jsonl", [record(title="first"), record(title="second")])
    _, report = select_inputs([raw], [])
    assert report["conflicts"][0]["reasons"] == ["ambiguous_records_at_winning_tier"]
    payload = record().to_dict()
    payload["uid"] = "forged"
    raw.write_text(json.dumps(payload) + "\nnot json\n", encoding="utf-8")
    rows, errors = read_inputs([raw])
    assert rows == []
    assert len(errors) == 2


def test_prepare_failure_preserves_existing_output(tmp_path):
    raw_dir, enriched_dir, output = [tmp_path / n for n in ("raw", "enriched", "selected")]
    save(raw_dir / "a.jsonl", [record(label="fake")])
    save(enriched_dir / "a.jsonl", [record(label="real")])
    output.mkdir()
    previous = output / "records.jsonl"
    previous.write_text("existing reviewed corpus\n", encoding="utf-8")
    result = script("prepare_inputs").main(["--raw-dir", str(raw_dir), "--enriched-dir",
                                           str(enriched_dir), "--out-dir", str(output)])
    assert result == 2
    assert previous.read_text() == "existing reviewed corpus\n"
    assert json.loads((output / "selection_report.json").read_text())["conflicts"]


def test_audit_detects_content_duplicates_and_split_leakage(tmp_path):
    (tmp_path / "one.png").write_bytes(b"same image bytes")
    (tmp_path / "two.png").write_bytes(b"same image bytes")
    first = record("1", label="fake", title="Fake investment scheme",
                   images=[ImageAsset(url="https://img/1", local_path="one.png",
                                      ocr_text="Guaranteed profit")])
    second = record("2", label="real", title="Official investment campaign",
                    images=[ImageAsset(url="https://img/2", local_path="two.png")])
    first.raw["cluster_id"] = second.raw["cluster_id"] = "same-claim"
    manifest = {"splits": {first.uid: "train", second.uid: "test"},
                "clusters": {first.uid: "same-claim", second.uid: "same-claim"}}
    report, review = audit_records([first, second], tmp_path, manifest)
    assert len(report["exact_image_duplicate_groups"]) == 1
    assert "cluster_overlap_across_splits" in report["split_checks"]["problems"]
    assert "exact_image_overlap_across_splits" in report["split_checks"]["problems"]
    assert report["images"]["records_with_eligible_ocr"] == 1
    assert len(review) == 2
    assert all("exact_image_shared_across_records" in r["reasons"] for r in review)
    assert first.label == "fake" and second.label == "real"


def test_missing_images_excluded_ocr_and_task_flags_are_review_only(tmp_path):
    rec = record(label="unknown", title="Political cheque allegation",
                 images=[ImageAsset(url="https://img/missing", local_path="missing.png"),
                         ImageAsset(url="https://img/outside", local_path="../secret.png"),
                         ImageAsset(url="https://img/stamp", role="annotated", ocr_text="FALSE")])
    report, rows = audit_records([rec, rec], tmp_path)
    reasons = rows[0]["reasons"]
    assert "missing_image_file" in reasons
    assert "image_path_outside_data_dir" in reasons
    assert "excluded_image_contains_ocr" in reasons
    assert "no_clear_investment_terms_in_claim_or_image" in reasons
    assert "possible_nonpromotion_finance_or_political_story" in reasons
    assert report["duplicate_uids"] == {rec.uid: 2}
    assert report["labels"] == {"unknown": 2}
    assert rows[0]["reviewer_label"] == ""
    assert rows[0]["image_only_claim_review"] == ""


def test_manual_export_preserves_text_and_has_blank_decisions(tmp_path):
    rec = record(title="=HYPERLINK(\"https://example.org\")", claim_text="निवेश करें")
    _, rows = audit_records([rec], tmp_path)
    write_review(rows, tmp_path / "review")
    with (tmp_path / "review/manual_review.csv").open(encoding="utf-8-sig", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert csv_rows[0]["title"].startswith("'=")
    assert csv_rows[0]["claim_text"] == "निवेश करें"
    assert csv_rows[0]["reviewer_label"] == ""
    lossless = json.loads((tmp_path / "review/manual_review.jsonl").read_text(encoding="utf-8"))
    assert lossless["title"] == rec.title
    assert lossless["uid"] == rec.uid


def test_stale_manifest_reports_missing_extra_and_hash(tmp_path):
    rec = record(label="real")
    report, _ = audit_records([rec], tmp_path,
                              {"splits": {"other": "unknown-split"}, "assignment_hash": "stale"})
    split = report["split_checks"]
    assert split["missing_uids"] == [rec.uid]
    assert split["extra_uids"] == ["other"]
    assert "invalid_split_names" in split["problems"]
    assert "manifest_assignment_hash_mismatch" in split["problems"]


def test_geometry_inspection_repairs_copies_and_preserves_roles_and_errors(tmp_path):
    from PIL import Image

    Image.new("RGB", (256, 128)).save(tmp_path / "hero.png")
    Image.new("RGB", (100, 67)).save(tmp_path / "sidebar.png")
    (tmp_path / "broken.png").write_bytes(b"not an image")
    rec = record(images=[
        ImageAsset(url="https://img/hero", local_path="hero.png", role="unknown"),
        ImageAsset(url="https://img/sidebar", local_path="sidebar.png", width=811, height=431),
        ImageAsset(url="https://img/missing", local_path="missing.png"),
        ImageAsset(url="https://img/broken", local_path="broken.png"),
    ])
    original = rec.to_dict()
    copies, report = inspect_image_dimensions([rec], tmp_path)
    assert rec.to_dict() == original
    assert [(a.width, a.height) for a in copies[0].images[:2]] == [(256, 128), (100, 67)]
    assert copies[0].images[0].role == "unknown"
    assert len(copies[0].images) == 4
    assert report["counts"] == {"missing": 1, "repaired": 2, "unreadable": 1}
    assert report["images"][1]["previous"] == {"width": 811, "height": 431}


def test_prepare_geometry_opt_in_keeps_source_file_unchanged(tmp_path):
    from PIL import Image

    Image.new("RGB", (320, 160)).save(tmp_path / "hero.png")
    rec = record(images=[ImageAsset(url="https://img/1", local_path="hero.png")])
    source = save(tmp_path / "raw/input.jsonl", [rec])
    original = source.read_bytes()
    out = tmp_path / "selected"
    result = script("prepare_inputs").main([
        "--raw-dir", str(source.parent), "--enriched-dir", str(tmp_path / "enriched"),
        "--out-dir", str(out), "--data-dir", str(tmp_path), "--inspect-images"])
    assert result == 0 and source.read_bytes() == original
    selected = json.loads((out / "records.jsonl").read_text())
    assert selected["images"][0]["width"] == 320
    assert selected["images"][0]["height"] == 160
    inspection = json.loads((out / "selection_report.json").read_text())["image_inspection"]
    assert inspection["counts"]["repaired"] == 1


def test_unfilled_automatic_review_queues_can_be_regenerated(tmp_path):
    _, rows = audit_records([record()], tmp_path)
    write_review(rows, tmp_path / "audit")
    before = {p.name: p.read_bytes() for p in (tmp_path / "audit").iterdir()}
    write_review(rows, tmp_path / "audit")
    assert {p.name: p.read_bytes() for p in (tmp_path / "audit").iterdir()} == before


@pytest.mark.parametrize("format_name", ["csv", "jsonl"])
def test_populated_review_queue_prevents_overwrite_of_all_audit_artifacts(tmp_path, format_name,
                                                                         capsys):
    rec = record()
    source = save(tmp_path / "input.jsonl", [rec])
    _, rows = audit_records([rec], tmp_path)
    output = tmp_path / "audit"
    write_review(rows, output)
    if format_name == "csv":
        csv_path = output / "manual_review.csv"
        with csv_path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            columns, entries = reader.fieldnames, list(reader)
        entries[0]["reviewer_ocr_quality"] = "Manually corrected Hindi text"
        with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(entries)
    else:
        entries = rows.copy()
        entries[0]["image_only_claim_review"] = "yes"
        (output / "manual_review.jsonl").write_text(json.dumps(entries[0]) + "\n", encoding="utf-8")
    (output / "dataset_audit.json").write_text('{"existing_report": true}\n', encoding="utf-8")
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError, match="human review entries"):
        write_review([], output)
    result = script("08_audit_dataset").main([
        "--in-file", str(source), "--data-dir", str(tmp_path), "--out-dir", str(output)])
    assert result == 2
    assert "Use a new --out-dir" in capsys.readouterr().err
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before


def test_unreadable_existing_review_queue_is_preserved(tmp_path):
    output = tmp_path / "audit"
    output.mkdir()
    malformed = output / "manual_review.jsonl"
    malformed.write_text("not JSON but possibly human work\n", encoding="utf-8")
    before = malformed.read_bytes()
    with pytest.raises(ValueError, match="Refusing to overwrite"):
        write_review([], output)
    assert malformed.read_bytes() == before
    assert not (output / "manual_review.csv").exists()
