"""Apply attributed, explicit human decisions to a separate candidate corpus."""

from __future__ import annotations

import csv
import hashlib
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

from PIL import Image

from ..schema.records import FactCheckRecord


def read_reviews(path: str | Path) -> tuple[list[dict], str]:
    """Return CSV rows and the hash of the exact review artifact."""
    path = Path(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"uid", "reviewer_task_relevant", "reviewer_label", "reviewer_image_usable",
                    "reviewer_image_path", "reviewer_name"}
        if not required.issubset(reader.fieldnames or []):
            missing = sorted(required - set(reader.fieldnames or []))
            raise ValueError(f"review CSV is missing columns: {', '.join(missing)}")
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError("review CSV contains rows with more values than header columns")
    return rows, digest


def apply_reviews(records, reviews, data_dir: str | Path,
                  csv_sha256: str, records_sha256: str):
    """Accept only complete yes/yes decisions, preserving source evidence in raw.

    Rows marked no, pending rows and conflicts remain outside the accepted
    corpus. Rejected rows and unreviewed candidate UIDs are explicitly reported.
    The original records and review queue are never mutated.
    """
    records = list(records)
    data_dir = Path(data_dir).resolve()
    candidates = defaultdict(list)
    for rec in records:
        candidates[rec.uid].append(rec)
    counts = Counter((row.get("uid") or "").strip() for row in reviews)
    accepted, decisions = [], []
    for line, row in enumerate(reviews, 2):
        uid = (row.get("uid") or "").strip()
        decision = {"csv_line": line, "uid": uid, "status": "excluded", "reasons": []}
        reasons = decision["reasons"]
        if not uid or uid not in candidates:
            reasons.append("unknown_uid")
        if counts[uid] > 1:
            reasons.append("duplicate_review_uid")
        if uid in candidates and len(candidates[uid]) > 1:
            reasons.append("duplicate_candidate_uid")
        values = {key: (row.get(key) or "").strip() for key in (
            "reviewer_task_relevant", "reviewer_label", "reviewer_image_usable", "reviewer_name",
            "reviewer_image_path", "reviewer_caption", "reviewer_caption_is_original",
            "reviewer_caption_source", "reviewer_notes", "image_only_claim_review",
            "reviewer_ocr_quality")}
        relevant = values["reviewer_task_relevant"].lower()
        usable = values["reviewer_image_usable"].lower()
        label = values["reviewer_label"].lower()
        if relevant == "no":
            reasons.append("reviewer_marked_irrelevant")
        elif relevant != "yes":
            reasons.append("pending_or_invalid_task_relevance")
        if usable == "no":
            reasons.append("reviewer_marked_image_unusable")
        elif usable != "yes":
            reasons.append("pending_or_invalid_image_decision")
        if label not in {"fake", "real"}:
            reasons.append("pending_or_invalid_label")
        if not values["reviewer_name"]:
            reasons.append("reviewer_name_required")
        if not values["reviewer_image_path"]:
            reasons.append("explicit_reviewed_image_path_required")
        if values["reviewer_caption"] and (
            values["reviewer_caption_is_original"].lower() != "yes"
            or not values["reviewer_caption_source"]
        ):
            reasons.append("caption_requires_original_confirmation_and_source")
        if reasons:
            decisions.append(decision)
            continue
        original = candidates[uid][0]
        selected_path = (data_dir / values["reviewer_image_path"]).resolve()
        if not selected_path.is_relative_to(data_dir):
            reasons.append("reviewed_image_outside_data_dir")
        matching = [a for a in original.images if a.local_path
                    and (data_dir / a.local_path).resolve() == selected_path]
        if len(matching) != 1:
            reasons.append("reviewed_image_must_match_exactly_one_candidate_asset")
        if not selected_path.is_file():
            reasons.append("reviewed_image_file_missing")
        if matching and matching[0].role in {"annotated", "site_chrome"}:
            reasons.append("reviewed_image_is_annotation_or_site_chrome")
        if reasons:
            decisions.append(decision)
            continue
        try:
            image_bytes = selected_path.read_bytes()
            image_digest = hashlib.sha256(image_bytes).hexdigest()
            if matching[0].sha256 and matching[0].sha256.lower() != image_digest:
                reasons.append("reviewed_image_hash_mismatch")
            with Image.open(BytesIO(image_bytes)) as decoded:
                width, height = decoded.size
                decoded.verify()
            # verify() checks the container; a second open/load forces actual
            # pixel decoding so truncated compressed streams cannot be accepted.
            with Image.open(BytesIO(image_bytes)) as decoded:
                decoded.load()
        except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
            reasons.append("reviewed_image_file_invalid")
            decision["image_error"] = str(exc)
        if reasons:
            decisions.append(decision)
            continue
        rec = FactCheckRecord.from_dict(original.to_dict())
        chosen = next(a for a in rec.images if a.local_path == matching[0].local_path)
        provenance = {
            "method": "csv_review_import_v1", "uid": uid,
            "task_relevant": True, "image_usable": True, "label": label,
            "image_path": chosen.local_path, "reviewer_name": values["reviewer_name"],
            "reviewer_notes": values["reviewer_notes"], "csv_line": line,
            "reviewer_ocr_quality": values["reviewer_ocr_quality"],
            "image_sha256": image_digest,
            "image_dimensions": {"width": width, "height": height},
            "review_csv_sha256": csv_sha256, "source_records_sha256": records_sha256,
            "original_title": rec.title, "original_claim_text": rec.claim_text,
            "original_images": [a.to_dict() for a in rec.images],
            "original_caption": values["reviewer_caption"],
            "caption_source": values["reviewer_caption_source"],
            "image_only_claim_review": values["image_only_claim_review"],
        }
        # Source descriptions/verdict-bearing titles are preserved above; only
        # the explicitly confirmed original caption enters the text branch.
        rec.title = ""
        rec.claim_text = values["reviewer_caption"]
        chosen.role = "creative"
        chosen.sha256 = image_digest
        chosen.width, chosen.height = width, height
        rec.images = [chosen]
        rec.raw["manual_review"] = provenance
        rec.label = label
        rec.is_finance = True
        accepted.append(rec)
        decision["status"] = "accepted"
        decisions.append(decision)
    report = {
        "schema_version": 1, "source_candidate_rows": len(records),
        "review_rows": len(reviews), "accepted_records": len(accepted),
        "accepted_labels": dict(sorted(Counter(r.label for r in accepted).items())),
        "review_csv_sha256": csv_sha256, "source_records_sha256": records_sha256,
        "decisions": decisions,
        "unreviewed_uids": sorted(set(candidates) - set(counts)),
        "notice": "Only explicit attributed decisions are imported. No reviewed subset manifest "
                  "or automatic reviewed:true assertion was generated.",
    }
    return sorted(accepted, key=lambda r: r.uid), report
