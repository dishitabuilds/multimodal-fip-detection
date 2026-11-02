#!/usr/bin/env python3
"""Build an explicitly provisional AI-assisted corpus for internal smoke runs.

This never creates human-review approvals. A manifest records each selected
existing asset, its byte hash, source evidence and a proposed label. The result
must not be used as the project's final research evaluation dataset.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.schema.records import FactCheckRecord  # noqa: E402
from fipd.utils.io import read_records  # noqa: E402


def build_pilot(manifest, data_dir):
    data_dir = Path(data_dir).resolve()
    if (manifest.get("reviewer_kind") != "ai_assisted"
            or manifest.get("intended_use") != "internal_smoke_only"):
        raise ValueError("only explicitly AI-assisted internal-smoke manifests are supported")
    records, seen, sources, groups = [], set(), {}, defaultdict(set)
    for item in manifest.get("records", []):
        uid, label = item["uid"], item["suggested_label"]
        if uid in seen or label not in {"real", "fake"}:
            raise ValueError("duplicate UID or invalid proposed label")
        seen.add(uid)
        for key in ("scope_evidence", "source_label_evidence", "cluster_id"):
            if not isinstance(item.get(key), str) or not item[key].strip():
                raise ValueError(f"per-item {key} is required")
        source = Path(item["source_records"]).resolve()
        if not source.is_relative_to(data_dir):
            raise ValueError("source records must be within data-dir")
        if source not in sources:
            candidates = list(read_records(source, strict=True))
            sources[source] = (candidates, hashlib.sha256(source.read_bytes()).hexdigest())
        matches = [r for r in sources[source][0] if r.uid == uid]
        if len(matches) != 1:
            raise ValueError(f"UID {uid} must have exactly one source record")
        rec = FactCheckRecord.from_dict(matches[0].to_dict())
        path = (data_dir / item["selected_image_path"]).resolve()
        if not path.is_relative_to(data_dir):
            raise ValueError("selected image is outside data-dir")
        assets = [a for a in rec.images if a.local_path
                  and (data_dir / a.local_path).resolve() == path]
        if len(assets) != 1 or assets[0].role in {"annotated", "site_chrome"}:
            raise ValueError("selected asset is absent, ambiguous or already excluded")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item.get("image_sha256"):
            raise ValueError("selected image differs from the visually inspected byte hash")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
        selected = assets[0]
        rec.raw["ai_assisted_review"] = {
            **item,
            "reviewer_kind": "ai_assisted", "intended_use": "internal_smoke_only",
            "human_approved": False, "source_records_sha256": sources[source][1],
            "source_url": rec.url, "source_verdict": rec.verdict_raw,
            "original_title": rec.title, "original_claim_text": rec.claim_text,
            "original_assets": [a.to_dict() for a in rec.images],
        }
        rec.raw["cluster_id"] = item["cluster_id"]
        rec.title = rec.claim_text = rec.verdict_raw = ""
        selected.role = "creative"
        selected.width, selected.height, selected.sha256 = width, height, digest
        # Re-OCR the selected image in its own output/cache directory.
        selected.ocr_text = None
        rec.images, rec.label, rec.is_finance = [selected], label, True
        records.append(rec)
        groups[label].add(item["cluster_id"])
    if any(len(groups[label]) < 3 for label in ("fake", "real")):
        raise ValueError("the smoke pilot needs at least three proposed groups per class")
    return sorted(records, key=lambda r: r.uid), {
        "reviewer_kind": "ai_assisted", "intended_use": "internal_smoke_only",
        "human_approved": False, "records": len(records),
        "proposed_labels": dict(Counter(r.label for r in records)),
        "notice": "Provisional labels, source/genre shortcuts and tiny sample; no research accuracy claim.",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", required=True, help="new output directory")
    args = parser.parse_args(argv)
    output = Path(args.out_dir)
    if output.exists():
        parser.error("--out-dir must be new; existing artifacts are preserved")
    try:
        manifest_path = Path(args.manifest)
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        records, report = build_pilot(manifest, args.data_dir)
        report["selection_manifest_sha256"] = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    output.mkdir(parents=True)
    (output / "records.jsonl").write_text("".join(r.to_json() + "\n" for r in records), encoding="utf-8")
    (output / "pilot_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Provisional internal smoke: {report['records']} records {report['proposed_labels']} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
