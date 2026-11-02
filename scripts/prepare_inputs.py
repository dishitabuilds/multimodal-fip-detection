#!/usr/bin/env python3
"""Select one record per UID, explicitly preferring enriched over raw inputs.

    python scripts/prepare_inputs.py
    python scripts/prepare_inputs.py --enriched-file data/raw/gfc_enriched.jsonl

Outputs records.jsonl and selection_report.json under data/selected_input.
Conflicts/input errors produce a report and exit 2 without replacing records.
For new stage-02 runs, prefer --out-file data/enriched/googlefactcheck.jsonl.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.datasets.audit import inspect_image_dimensions, jsonl_paths, select_inputs  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", default="data/raw")
    parser.add_argument("--enriched-dir", default="data/enriched")
    parser.add_argument("--enriched-file", action="append", default=[],
                        help="explicit enriched file, excluded from raw tier; repeatable")
    parser.add_argument("--out-dir", default="data/selected_input")
    parser.add_argument("--data-dir", default="data", help="root for data-relative image paths")
    parser.add_argument("--inspect-images", action="store_true",
                        help="refresh decoded dimensions on selected copies; report missing/corrupt assets")
    args = parser.parse_args(argv)
    explicit = {Path(p).resolve() for p in args.enriched_file}
    raw = [p for p in jsonl_paths(args.raw_dir) if p.resolve() not in explicit]
    enriched = jsonl_paths(args.enriched_dir) + list(explicit)
    selected, report = select_inputs(raw, enriched)
    if args.inspect_images and report["safe_to_write"]:
        selected, inspection = inspect_image_dimensions(selected, args.data_dir)
        report["image_inspection"] = inspection
    out_dir = Path(args.out_dir).resolve()
    if out_dir in {Path(args.raw_dir).resolve(), Path(args.enriched_dir).resolve()}:
        parser.error("output directory must differ from input directories")
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "selection_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    if not report["safe_to_write"]:
        print(f"Selection blocked: {len(report['conflicts'])} conflicts, "
              f"{len(report['input_errors'])} input errors; proposed {len(selected)} records.")
        print(f"Existing records.jsonl was not changed. Review {report_path}")
        return 2
    target = out_dir / "records.jsonl"
    temporary = target.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(r.to_json() + "\n" for r in selected), encoding="utf-8")
    temporary.replace(target)
    print(f"Selected {len(selected)} records from {report['input_rows']} rows -> {target}")
    print(f"Provenance and duplicate decisions: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
