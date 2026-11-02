"""Strict experiment input validation shared by neural and classical models."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path

from .examples import to_examples
from ..schema.records import FactCheckRecord, VALID_LABELS


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_experiment_inputs(records_file: Path, manifest_file: Path, text_cfg,
                           allow_internal_smoke: bool = False,
                           require_manual_review: bool = False):
    """Recheck the current corpus, rather than trust a historical leakage flag.

    Collection helpers intentionally skip malformed rows. Experiments must fail
    closed instead: every input row and UID needs a verified split assignment.
    """
    records = []
    seen = set()
    with records_file.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("record must be a JSON object")
                rec = FactCheckRecord.from_dict(payload)
                for name in ("source", "source_id", "url", "title", "claim_text"):
                    if not isinstance(getattr(rec, name), str):
                        raise ValueError(f"{name} must be a string")
                if not isinstance(rec.label, str) or rec.label not in VALID_LABELS:
                    raise ValueError("label must be real, fake or unknown")
                if not isinstance(rec.raw, dict):
                    raise ValueError("raw must be an object")
                assisted = rec.raw.get("ai_assisted_review")
                if assisted and not allow_internal_smoke:
                    raise ValueError("provisional AI-assisted labels require an explicit "
                                     "internal_smoke_only configuration; they are not research data")
                if require_manual_review:
                    review = rec.raw.get("manual_review") or {}
                    if (review.get("task_relevant") is not True
                            or review.get("image_usable") is not True
                            or review.get("label") != rec.label
                            or not review.get("reviewer_name")):
                        raise ValueError("lab research data requires attributed manual review; "
                                         "use scripts/apply_reviews.py, or --internal-pilot "
                                         "only for an unvalidated engineering run")
                cluster = rec.raw.get("cluster_id")
                if cluster is not None and not isinstance(cluster, str):
                    raise ValueError("raw.cluster_id must be a string")
                for asset in rec.images:
                    if asset.local_path is not None and not isinstance(asset.local_path, str):
                        raise ValueError("image local_path must be a string or null")
                    if not isinstance(asset.role, str):
                        raise ValueError("image role must be a string")
                    if asset.ocr_text is not None and not isinstance(asset.ocr_text, str):
                        raise ValueError("image ocr_text must be a string or null")
                    for name in ("width", "height"):
                        value = getattr(asset, name)
                        if value is not None and (type(value) is not int or value < 0):
                            raise ValueError(f"image {name} must be a nonnegative integer or null")
                if "uid" in payload and payload["uid"] != rec.uid:
                    raise ValueError("serialized UID does not match record source/source_id")
                if rec.uid in seen:
                    raise ValueError(f"duplicate record UID: {rec.uid}")
                seen.add(rec.uid)
                records.append(rec)
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                raise ValueError(f"{records_file}:{line_number}: invalid record ({exc})") from exc
    if not records:
        raise ValueError("no records to train; run collection and curation first")

    meta = json.loads(manifest_file.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError("split manifest must be an object")
    assignments, clusters = meta.get("splits"), meta.get("clusters")
    if not isinstance(assignments, dict) or not isinstance(clusters, dict):
        raise ValueError("split manifest must include splits and clusters mappings")
    for name, mapping in (("splits", assignments), ("clusters", clusters)):
        missing, extra = seen - mapping.keys(), mapping.keys() - seen
        if missing or extra:
            raise ValueError(f"manifest {name} UID coverage differs from current records; "
                             f"missing={sorted(missing)}, extra={sorted(extra)}")
    if any(value not in ("train", "val", "test") for value in assignments.values()):
        raise ValueError("manifest contains an invalid split name")
    if any(not isinstance(value, str) or not value for value in clusters.values()):
        raise ValueError("manifest cluster IDs must be non-empty strings")
    current_clusters = {rec.uid: rec.raw.get("cluster_id") or rec.uid for rec in records}
    owners = {}
    for uid, cluster in current_clusters.items():
        split = assignments[uid]
        if cluster in owners and owners[cluster] != split:
            raise ValueError(f"current claim cluster {cluster} spans splits; regenerate manifest")
        owners[cluster] = split
    stale = [uid for uid in seen if clusters[uid] != current_clusters[uid]]
    if stale:
        raise ValueError(f"manifest cluster assignments differ from current records: {sorted(stale)}")
    calculated_hash = hashlib.sha256(
        json.dumps(sorted(assignments.items()), sort_keys=True).encode()).hexdigest()[:16]
    if meta.get("assignment_hash") != calculated_hash:
        raise ValueError("manifest assignment hash does not match its split assignments")
    expected_counts = dict(Counter(assignments.values()))
    expected_cluster_counts = dict(Counter(owners.values()))
    if (type(meta.get("n_records")) is not int or meta["n_records"] != len(records)
            or type(meta.get("n_clusters")) is not int or meta["n_clusters"] != len(owners)
            or meta.get("counts") != expected_counts
            or meta.get("cluster_counts") != expected_cluster_counts):
        raise ValueError("manifest record/cluster counts do not match current assignments")
    if meta.get("leakage_ok") is not True:
        raise ValueError("manifest records a failed or missing leakage check")

    by_split = {name: [] for name in ("train", "val", "test")}
    for example in to_examples(records, assignments, text_cfg):
        by_split[example.split].append(example)
    return by_split, meta


def common_cohort(splits, image_root: Path, require_image: bool = True):
    """Filter once, before arm selection; missing images cannot favor one arm."""
    from PIL import Image

    cohort, excluded, signatures = {}, {}, {}
    checked_images, image_owners = {}, {}
    seen = set()
    for split in ("train", "val", "test"):
        cohort[split], excluded[split], signatures[split] = [], [], []
        for ex in splits.get(split, []):
            if ex.uid in seen:
                raise ValueError(f"duplicate UID across cohort/splits: {ex.uid}")
            seen.add(ex.uid)
            path = image_root / ex.image_path if ex.image_path else None
            image_digest = None
            if path is not None:
                key = str(path.resolve())
                if key not in checked_images:
                    try:
                        with Image.open(path) as image:
                            image.convert("RGB").load()
                        checked_images[key] = file_hash(path)
                    except (OSError, ValueError):
                        checked_images[key] = None
                image_digest = checked_images[key]
            if not ex.text.strip() or (require_image and image_digest is None):
                excluded[split].append(ex.uid)
                continue
            owner = image_owners.get(image_digest) if image_digest else None
            if owner is not None and owner[0] != split:
                raise ValueError(f"chosen image bytes are shared across splits: "
                                 f"{owner[1]} ({owner[0]}) and {ex.uid} ({split}); "
                                 "regenerate leakage-safe splits before training")
            if image_digest:
                image_owners[image_digest] = (split, ex.uid)
            cohort[split].append(ex)
            signatures[split].append({"uid": ex.uid, "label": ex.label,
                                      "text": ex.text, "image_sha256": image_digest})
        labels = Counter(ex.label for ex in cohort[split])
        if set(labels) != {0, 1}:
            raise ValueError(f"{split} must contain both real (0) and fake (1) after "
                             f"common input filtering; got {dict(labels)}. "
                             "Review the data and regenerate the splits before training.")
    return cohort, excluded, fingerprint(signatures)
