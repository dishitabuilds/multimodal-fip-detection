"""Offline input selection and dataset inspection; heuristics never assign labels.

Selection gives explicit enriched inputs precedence over raw inputs. Ambiguous
records are reported and excluded from the proposed selection, so callers can
refuse publication without losing the original files. Audit exports retain every
input row and leave all human review decisions blank.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urldefrag

from ..curation.labels import map_verdict
from ..schema.records import FactCheckRecord, LABEL_FAKE, LABEL_REAL, LABEL_UNKNOWN

KNOWN_LABELS = {LABEL_FAKE, LABEL_REAL}
SPLITS = ("train", "val", "test")
_LABEL_MARKERS = re.compile(
    r"\b(?:fake|false|misleading|debunked|doctored|fact[ -]?check|genuine|"
    r"official|legitimate|authentic|verified|not true)\b|"
    r"(?:फर्जी|फ़र्ज़ी|भ्रामक|झूठ|असत्य)", re.I,
)
_INVESTMENT = re.compile(
    r"\b(?:invest\w*|trad(?:e|ing|er)|stocks?|shares?|mutual funds?|"
    r"returns?|profits?|ponzi|crypto\w*|forex|demat|broker\w*|"
    r"money doubl\w*|paisa double)\b|(?:निवेश|मुनाफा|शेयर|पैसा डबल|रिटर्न)", re.I,
)
_OTHER_FINANCE = re.compile(
    r"\b(?:tax slabs?|crowdfund\w*|election\w*|congress|bjp|cash seiz\w*|"
    r"seiz(?:e|ed|ure)|cheques?|temple|mosque)\b", re.I,
)


@dataclass
class InputRow:
    record: FactCheckRecord
    path: str
    line: int
    tier: str

    def origin(self) -> dict[str, Any]:
        return {"path": self.path, "line": self.line, "tier": self.tier}


def jsonl_paths(directory: str | Path) -> list[Path]:
    """Immediate JSONL files only, matching the stage scripts' reader."""
    return sorted(Path(directory).glob("*.jsonl"))


def read_inputs(paths: Iterable[str | Path], tier: str = "audit") -> tuple[list[InputRow], list]:
    """Read strictly, preserving origin and reporting malformed or forged UIDs."""
    rows, errors = [], []
    for path in sorted({Path(p).resolve() for p in paths}, key=str):
        try:
            lines = path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeError) as exc:
            errors.append({"path": str(path), "error": str(exc)})
            continue
        for line_number, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                rec = FactCheckRecord.from_dict(data)
                if not rec.source or not rec.source_id or not rec.url:
                    raise ValueError("source, source_id and url must be nonempty")
                if rec.label not in KNOWN_LABELS | {LABEL_UNKNOWN}:
                    raise ValueError(f"invalid label: {rec.label!r}")
                if data.get("uid", rec.uid) != rec.uid:
                    raise ValueError("stored UID does not match source/source_id")
                rows.append(InputRow(rec, str(path), line_number, tier))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                errors.append({"path": str(path), "line": line_number, "error": str(exc)})
    return rows, errors


def select_inputs(raw_paths: Iterable[str | Path], enriched_paths: Iterable[str | Path]):
    """Return a proposed selection and report; any conflict blocks publication.

    An enriched record replaces its raw counterpart as a whole. Conflicting
    known labels/verdicts, URLs, or distinct records at the winning tier require
    review. Unknown labels are never interpreted as real or guessed.
    """
    raw, raw_errors = read_inputs(raw_paths, "raw")
    enriched, enriched_errors = read_inputs(enriched_paths, "enriched")
    groups = defaultdict(list)
    for row in raw + enriched:
        groups[row.record.uid].append(row)
    selected, selections, conflicts, duplicate_uids = [], {}, [], {}
    for uid, variants in sorted(groups.items()):
        if len(variants) > 1:
            duplicate_uids[uid] = [r.origin() for r in variants]
        labels = {r.record.label for r in variants if r.record.label in KNOWN_LABELS}
        verdicts = {map_verdict(r.record.verdict_raw) for r in variants} & KNOWN_LABELS
        urls = {urldefrag(r.record.url)[0].rstrip("/") for r in variants}
        reasons = []
        if len(labels | verdicts) > 1:
            reasons.append("conflicting_labels_or_verdicts")
        if len(urls) > 1:
            reasons.append("conflicting_provenance_urls")
        winners = [r for r in variants if r.tier == "enriched"] or variants
        signatures = set()
        for row in winners:
            payload = row.record.to_dict()
            payload.pop("collected_at", None)
            signatures.add(json.dumps(payload, sort_keys=True, ensure_ascii=False))
        if len(signatures) > 1:
            reasons.append("ambiguous_records_at_winning_tier")
        winner = sorted(winners, key=lambda r: (r.path, r.line))[0]
        if labels and winner.record.label == LABEL_UNKNOWN:
            reasons.append("enrichment_would_discard_known_label")
        if reasons:
            conflicts.append({"uid": uid, "reasons": reasons,
                              "labels": sorted(labels), "verdict_labels": sorted(verdicts),
                              "variants": [r.origin() | {"source": r.record.source,
                                           "source_id": r.record.source_id,
                                           "url": r.record.url, "label": r.record.label,
                                           "verdict_raw": r.record.verdict_raw} for r in variants]})
            continue
        selected.append(winner.record)
        selections[uid] = {"selected": winner.origin(),
                           "alternatives": [r.origin() for r in variants if r is not winner]}
    report = {
        "schema_version": 1,
        "policy": "enriched replaces raw; unequal winning-tier records require review",
        "input_rows": len(raw) + len(enriched), "unique_uids": len(groups),
        "selected_records": len(selected), "duplicate_uids": duplicate_uids,
        "conflicts": conflicts, "input_errors": raw_errors + enriched_errors,
        "selections": selections,
        "safe_to_write": bool(selected) and not conflicts and not raw_errors and not enriched_errors,
    }
    return selected, report


def _image_path(data_dir: Path, local_path: str) -> Path | None:
    path = (data_dir / local_path).resolve()
    return path if path.is_relative_to(data_dir.resolve()) else None


def inspect_image_dimensions(records: Iterable[FactCheckRecord], data_dir: str | Path):
    """Refresh decoded geometry on copies; retain all assets and their roles.

    Incorrect declared areas can cause the loader to choose article sidebar art
    over the intended hero. This opt-in inspection repairs geometry only; it
    cannot establish which asset depicts the investment claim.
    """
    from PIL import Image  # lazy: input selection itself needs no image decoder

    copies = [FactCheckRecord.from_dict(rec.to_dict()) for rec in records]
    evidence, counts = [], Counter()
    for rec in copies:
        for asset in rec.images:
            previous = {"width": asset.width, "height": asset.height}
            result = {"uid": rec.uid, "path": asset.local_path, "previous": previous}
            if not asset.local_path:
                result["status"] = "not_downloaded"
            else:
                path = _image_path(Path(data_dir), asset.local_path)
                if path is None:
                    result["status"] = "outside_data_dir"
                elif not path.is_file():
                    result["status"] = "missing"
                else:
                    try:
                        with Image.open(path) as decoded:
                            width, height = decoded.size
                            decoded.verify()
                        result["decoded"] = {"width": width, "height": height}
                        result["status"] = ("repaired" if (asset.width, asset.height) != (width, height)
                                            else "unchanged")
                        asset.width, asset.height = width, height
                    except (OSError, ValueError, SyntaxError, Image.DecompressionBombError) as exc:
                        result["status"] = "unreadable"
                        result["error"] = str(exc)
            counts[result["status"]] += 1
            evidence.append(result)
    return copies, {"counts": dict(sorted(counts.items())), "images": evidence,
                    "notice": "Geometry refreshed on selected copies only. All assets and roles "
                              "are retained; creative relevance still requires human review."}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_records(records: Iterable[FactCheckRecord], data_dir: str | Path,
                  manifest: dict | None = None, origins: dict | None = None):
    """Return a machine-readable report and one review row per candidate.

    Exact duplicates are checked from local bytes, independently of collector
    hashes. Missing paths are reported; paths outside data_dir are never read.
    Relevance and leakage findings are screening flags requiring human judgment.
    """
    records = sorted(records, key=lambda r: (r.uid, r.to_json()))
    data_dir = Path(data_dir)
    review, image_owners, path_cache = [], defaultdict(set), {}
    labels, sources, languages, issues, image_counts = [Counter() for _ in range(5)]
    uid_counts = Counter(r.uid for r in records)
    for rec in records:
        labels[rec.label] += 1
        sources[rec.source] += 1
        languages[rec.language or "unspecified"] += 1
        reasons, image_rows, ocr_parts = [], [], []
        usable_count = ocr_count = 0
        markers = {}
        for field, text in (("title", rec.title), ("claim_text", rec.claim_text)):
            hits = sorted(set(m.group(0).lower() for m in _LABEL_MARKERS.finditer(text)))
            if hits:
                markers[field] = hits
                reasons.append(f"possible_label_marker_in_{field}")
        seen_paths = set()
        for asset in rec.images:
            image_counts["assets"] += 1
            eligible = asset.role not in {"annotated", "site_chrome"}
            exists = False
            digest = None
            asset_reasons = []
            if not asset.local_path:
                asset_reasons.append("image_not_downloaded")
            else:
                image_counts["local_references"] += 1
                path = _image_path(data_dir, asset.local_path)
                if path is None:
                    asset_reasons.append("image_path_outside_data_dir")
                elif not path.is_file():
                    asset_reasons.append("missing_image_file")
                else:
                    exists = True
                    if path not in path_cache:
                        try:
                            path_cache[path] = (_sha256(path), path.stat().st_size)
                        except OSError:
                            path_cache[path] = (None, 0)
                    digest, size = path_cache[path]
                    if not digest or not size:
                        asset_reasons.append("empty_or_unreadable_image_file")
                    else:
                        image_owners[digest].add(rec.uid)
                        usable_count += eligible
                        if asset.sha256 and asset.sha256 != digest:
                            asset_reasons.append("stored_image_hash_mismatch")
                    if path in seen_paths:
                        asset_reasons.append("duplicate_image_reference")
                    seen_paths.add(path)
            if not eligible:
                asset_reasons.append("annotation_or_site_chrome")
            if asset.ocr_text and asset.ocr_text.strip():
                image_counts["assets_with_ocr"] += 1
                if eligible:
                    ocr_parts.append(asset.ocr_text)
                    ocr_count += 1
                else:
                    asset_reasons.append("excluded_image_contains_ocr")
                hits = sorted(set(m.group(0).lower()
                                  for m in _LABEL_MARKERS.finditer(asset.ocr_text)))
                if hits:
                    markers.setdefault("ocr_text", []).extend(hits)
                    reasons.append("possible_label_marker_in_ocr")
            elif eligible and exists:
                asset_reasons.append("eligible_image_without_ocr")
            for reason in asset_reasons:
                image_counts[reason] += 1
            reasons.extend(asset_reasons)
            image_rows.append({"path": asset.local_path, "url": asset.url, "role": asset.role,
                               "exists": exists, "sha256": digest,
                               "ocr_text": asset.ocr_text or "", "reasons": asset_reasons})
        if not usable_count:
            reasons.append("no_usable_local_image")
        if not ocr_count:
            reasons.append("no_eligible_image_ocr")
        if rec.label not in KNOWN_LABELS:
            reasons.append("label_needs_review")
        if uid_counts[rec.uid] > 1:
            reasons.append("duplicate_uid")
        task_text = " ".join([rec.title, rec.claim_text, *ocr_parts])
        if not _INVESTMENT.search(task_text):
            reasons.append("no_clear_investment_terms_in_claim_or_image")
        if _OTHER_FINANCE.search(task_text):
            reasons.append("possible_nonpromotion_finance_or_political_story")
        reasons = sorted(set(reasons))
        issues.update(reasons)
        review.append({
            "uid": rec.uid, "source": rec.source, "source_id": rec.source_id,
            "url": rec.url, "current_label": rec.label, "verdict_raw": rec.verdict_raw,
            "language": rec.language, "title": rec.title, "claim_text": rec.claim_text,
            "cluster_id": rec.raw.get("cluster_id") or rec.uid,
            "split": (manifest or {}).get("splits", {}).get(rec.uid, ""),
            "provenance": (origins or {}).get(rec.uid, {}), "images": image_rows,
            "leakage_markers": markers, "reasons": reasons,
            "reviewer_task_relevant": "", "reviewer_label": "",
            "reviewer_image_usable": "", "reviewer_ocr_quality": "",
            "image_only_claim_review": "", "reviewer_name": "", "reviewer_notes": "",
            "reviewer_image_path": "", "reviewer_caption": "",
            "reviewer_caption_is_original": "", "reviewer_caption_source": "",
        })
        image_counts["records_with_usable_image"] += bool(usable_count)
        image_counts["records_with_eligible_ocr"] += bool(ocr_count)
    duplicates = {key: sorted(value) for key, value in sorted(image_owners.items())
                  if len(value) > 1}
    for row in review:
        if any(row["uid"] in uids for uids in duplicates.values()):
            row["reasons"].append("exact_image_shared_across_records")
            issues["exact_image_shared_across_records"] += 1
    blockers = []
    provisional_ai = [r.uid for r in records if r.raw.get("ai_assisted_review")]
    if provisional_ai:
        blockers.append("provisional_ai_labels_not_research_ready")
    if not records:
        blockers.append("empty_corpus")
    if not KNOWN_LABELS.issubset(labels):
        blockers.append("both_classes_required")
    if labels.get(LABEL_UNKNOWN, 0):
        blockers.append("unresolved_labels")
    if issues["duplicate_uid"]:
        blockers.append("duplicate_uids")
    if image_counts["records_with_usable_image"] != len(records):
        blockers.append("some_records_lack_usable_images")
    if image_counts["records_with_eligible_ocr"] != len(records):
        blockers.append("some_records_lack_ocr")
    split_report = _audit_manifest(records, manifest, duplicates) if manifest is not None else None
    if split_report and not split_report["ok"]:
        blockers.append("split_checks_failed")
    report = {
        "schema_version": 1, "input_records": len(records), "unique_uids": len(uid_counts),
        "provisional_ai_labelled_records": len(provisional_ai),
        "labels": dict(sorted(labels.items())), "sources": dict(sorted(sources.items())),
        "languages": dict(sorted(languages.items())), "images": dict(sorted(image_counts.items())),
        "duplicate_uids": {u: n for u, n in sorted(uid_counts.items()) if n > 1},
        "exact_image_duplicate_groups": duplicates, "review_issue_counts": dict(sorted(issues.items())),
        "split_checks": split_report, "automatic_blockers": blockers,
        "manual_review_status": "not_established_by_automatic_audit",
        "heuristic_notice": "Flags screen for review; they do not establish labels, task relevance, "
                            "OCR accuracy or image readability. No candidate was removed.",
    }
    return report, review


def _audit_manifest(records, manifest, image_duplicates):
    splits, clusters = manifest.get("splits", {}), manifest.get("clusters", {})
    uids = {r.uid for r in records}
    problems = []
    missing, extra = sorted(uids - splits.keys()), sorted(splits.keys() - uids)
    if missing:
        problems.append("records_missing_from_manifest")
    if extra:
        problems.append("manifest_contains_absent_records")
    if any(s not in SPLITS for s in splits.values()):
        problems.append("invalid_split_names")
    cluster_splits, balance, source_balance = defaultdict(set), {}, {}
    for split in SPLITS:
        members = [r for r in records if splits.get(r.uid) == split]
        balance[split] = dict(sorted(Counter(r.label for r in members).items()))
        source_balance[split] = dict(sorted(Counter(r.source for r in members).items()))
        if not KNOWN_LABELS.issubset(balance[split]):
            problems.append(f"{split}_does_not_contain_both_classes")
    stale_clusters = []
    for rec in records:
        split = splits.get(rec.uid)
        actual = rec.raw.get("cluster_id") or rec.uid
        declared = clusters.get(rec.uid, actual)
        if str(declared) != str(actual):
            stale_clusters.append(rec.uid)
        if split:
            cluster_splits[str(actual)].add(split)
            cluster_splits[str(declared)].add(split)
    overlap = {c: sorted(s) for c, s in sorted(cluster_splits.items()) if len(s) > 1}
    if overlap:
        problems.append("cluster_overlap_across_splits")
    if stale_clusters:
        problems.append("manifest_cluster_assignments_differ_from_records")
    image_overlap = {digest: sorted({splits[u] for u in owners if u in splits})
                     for digest, owners in image_duplicates.items()
                     if len({splits[u] for u in owners if u in splits}) > 1}
    if image_overlap:
        problems.append("exact_image_overlap_across_splits")
    calculated = hashlib.sha256(json.dumps(sorted(splits.items()), sort_keys=True).encode()
                                ).hexdigest()[:16]
    if manifest.get("assignment_hash") and manifest["assignment_hash"] != calculated:
        problems.append("manifest_assignment_hash_mismatch")
    if manifest.get("n_records", len(uids)) != len(uids):
        problems.append("manifest_record_count_mismatch")
    return {"ok": not problems, "problems": problems, "missing_uids": missing,
            "extra_uids": extra, "label_balance": balance, "source_balance": source_balance,
            "cluster_overlap": overlap, "stale_cluster_uids": stale_clusters,
            "exact_image_overlap": image_overlap, "calculated_assignment_hash": calculated}


def protect_existing_reviews(out_dir: str | Path) -> None:
    """Preflight both queue formats before any existing audit artifact changes.

    Automatic queues with entirely blank decisions may be regenerated. A human
    entry in either format, or an unreadable existing queue, requires a new
    output directory so the review evidence cannot be silently replaced.
    """
    out_dir = Path(out_dir)

    def has_decision(row):
        return any((str(value).strip() if value is not None else "")
                   for key, value in row.items()
                   if isinstance(key, str)
                   and (key.startswith("reviewer_") or key == "image_only_claim_review"))

    for filename in ("manual_review.csv", "manual_review.jsonl"):
        path = out_dir / filename
        if not path.exists():
            continue
        try:
            if path.suffix == ".csv":
                with path.open(encoding="utf-8-sig", newline="") as handle:
                    reader = csv.DictReader(handle, strict=True)
                    if reader.fieldnames and "reviewer_label" not in reader.fieldnames:
                        raise ValueError("existing queue has an unrecognized header")
                    for row in reader:
                        if None in row:
                            raise ValueError("existing queue has malformed CSV rows")
                        if has_decision(row):
                            raise ValueError("existing queue contains human review entries")
            else:
                for line in path.read_text(encoding="utf-8-sig").splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("existing queue contains a non-object row")
                    if has_decision(row):
                        raise ValueError("existing queue contains human review entries")
        except (OSError, UnicodeError, ValueError, csv.Error) as exc:
            raise ValueError(f"Refusing to overwrite {path}: {exc}. "
                             "Use a new --out-dir; all existing audit artifacts were preserved.") from exc


def write_review(rows: list[dict], out_dir: str | Path) -> None:
    """Write blank queues while preserving any existing human review entries."""
    out_dir = Path(out_dir)
    protect_existing_reviews(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "manual_review.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    columns = ["uid", "source", "source_id", "url", "current_label", "verdict_raw", "language",
               "title", "claim_text", "cluster_id", "split", "image_paths", "ocr_text",
               "provenance", "reasons", "reviewer_task_relevant", "reviewer_label",
               "reviewer_image_usable", "reviewer_ocr_quality", "image_only_claim_review",
               "reviewer_image_path", "reviewer_caption", "reviewer_caption_is_original",
               "reviewer_caption_source",
               "reviewer_name", "reviewer_notes"]
    with (out_dir / "manual_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            flat = {key: row.get(key, "") for key in columns}
            flat["image_paths"] = " | ".join(a["path"] or "" for a in row["images"])
            flat["ocr_text"] = " | ".join(a["ocr_text"] for a in row["images"])
            flat["provenance"] = json.dumps(row["provenance"], ensure_ascii=False, sort_keys=True)
            flat["reasons"] = " | ".join(row["reasons"])
            for key, value in flat.items():
                if isinstance(value, str) and value.startswith(("=", "+", "-", "@", "\t", "\r")):
                    flat[key] = "'" + value
            writer.writerow(flat)
