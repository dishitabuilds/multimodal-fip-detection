#!/usr/bin/env python3
"""Audit local candidates and export a manual-review queue without changing them.

    python scripts/08_audit_dataset.py
    python scripts/08_audit_dataset.py --in-file data/selected_input/records.jsonl \
        --selection-report data/selected_input/selection_report.json

No OCR/model execution or network access. The JSON report records automatic
blockers; heuristic review flags never establish labels or remove candidates.
Exit 2 means malformed input was found, 0 means the audit completed (even when
the corpus is not ready). --fail-on-blockers also exits 2 for readiness blockers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.datasets.audit import audit_records, read_inputs, write_review  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-file", default="data/interim/finance.jsonl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--manifest", help="optional existing split_manifest.json to verify")
    parser.add_argument("--selection-report", help="optional provenance report from prepare_inputs.py")
    parser.add_argument("--out-dir", default="data/processed/audit")
    parser.add_argument("--fail-on-blockers", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8")) if args.manifest else None
    selection = (json.loads(Path(args.selection_report).read_text(encoding="utf-8"))
                 if args.selection_report else {})
    rows, errors = read_inputs([args.in_file])
    origins = selection.get("selections", {})
    if not origins:
        origins = {r.record.uid: {"input": r.origin()} for r in rows}
    report, review = audit_records([r.record for r in rows], args.data_dir, manifest, origins)
    report["input_file"] = str(Path(args.in_file).resolve())
    report["input_errors"] = errors
    if errors:
        report["automatic_blockers"].append("malformed_input_rows")
    out_dir = Path(args.out_dir)
    try:
        write_review(review, out_dir)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    report_path = out_dir / "dataset_audit.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(f"Audited {report['input_records']} records; labels: {report['labels']}")
    print(f"Automatic blockers: {', '.join(report['automatic_blockers']) or 'none detected'}")
    print(f"Report: {report_path.resolve()}")
    print(f"Review queue: {(out_dir / 'manual_review.csv').resolve()}")
    print("Review flags require human judgment. Candidate records were not changed.")
    return 2 if errors or (args.fail_on_blockers and report["automatic_blockers"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
