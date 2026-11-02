#!/usr/bin/env python3
"""Import explicit accepted review decisions into a new output directory.

Required CSV decisions: reviewer_task_relevant=yes, reviewer_label=fake/real,
reviewer_image_usable=yes, reviewer_name and reviewer_image_path (an existing
asset in that candidate). Optional reviewer_caption requires
reviewer_caption_is_original=yes and reviewer_caption_source. Pending, rejected
or conflicting rows are reported without changing the source corpus.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.datasets.audit import read_inputs  # noqa: E402
from fipd.datasets.reviews import apply_reviews, read_reviews  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--reviews", required=True)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", required=True, help="new directory; existing paths are refused")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir).resolve()
    if out_dir.exists():
        parser.error("--out-dir must be a new directory to preserve existing artifacts")
    rows, errors = read_inputs([args.records])
    if errors:
        parser.error(f"candidate input is invalid: {errors[0]}")
    try:
        reviews, csv_digest = read_reviews(args.reviews)
        records_digest = hashlib.sha256(Path(args.records).read_bytes()).hexdigest()
    except (OSError, UnicodeError, ValueError) as exc:
        parser.error(str(exc))
    accepted, report = apply_reviews([r.record for r in rows], reviews, args.data_dir,
                                     csv_digest, records_digest)
    report["source_records"] = str(Path(args.records).resolve())
    report["source_reviews"] = str(Path(args.reviews).resolve())
    out_dir.mkdir(parents=True)
    (out_dir / "review_import_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if accepted:
        (out_dir / "records.jsonl").write_text(
            "".join(rec.to_json() + "\n" for rec in accepted), encoding="utf-8")
    excluded = sum(r["status"] != "accepted" for r in report["decisions"])
    print(f"Imported {len(accepted)} explicit decisions; {excluded} rows excluded/pending. "
          f"Report: {out_dir / 'review_import_report.json'}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
