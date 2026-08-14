#!/usr/bin/env python3
"""Stage 04 — filter, label and deduplicate into a curated dataset.

Replaces the old build_dataset.py and adds the deduplication step, which the
earlier version was missing and without which every downstream metric is
suspect.

    python scripts/04_curate.py
    python scripts/04_curate.py --threshold 3.0        # loosen the filter
    python scripts/04_curate.py --require-image        # multimodal-ready only
    python scripts/04_curate.py --no-dedup             # skip clustering (fast)
    python scripts/04_curate.py --in-dir data/interim/ocr/easyocr   # post-OCR

Outputs into data/interim/:
    curated.jsonl        every record, annotated with label / finance / cluster
    finance.jsonl        the training candidate set (finance, labelled)
    canonical.jsonl      one representative per cluster
    curation_report.json counts for the report
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.curation.dedup import cluster_records, pick_canonical  # noqa: E402
from fipd.curation.finance_filter import score_record  # noqa: E402
from fipd.curation.labels import assign_label  # noqa: E402
from fipd.schema.records import LABEL_UNKNOWN  # noqa: E402
from fipd.utils.io import read_all, write_json, write_records  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--in-dir", default=None, help="default: <data-dir>/raw")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--require-image", action="store_true")
    ap.add_argument("--no-dedup", action="store_true")
    ap.add_argument("--image-threshold", type=int, default=8,
                    help="max Hamming distance for 'same image' (0-64)")
    ap.add_argument("--text-threshold", type=float, default=0.6,
                    help="min Jaccard on claim shingles for 'same claim'")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    log = setup(args.verbose)
    data_dir = Path(args.data_dir)
    in_dir = Path(args.in_dir) if args.in_dir else data_dir / "raw"
    interim = data_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    banner("Stage 04 — curate")

    records = read_all(in_dir)
    if not records:
        log.error("No records in %s — run 01_collect.py first", in_dir)
        return 1

    # ---- filter + label ------------------------------------------------
    per_source: dict[str, Counter] = defaultdict(Counter)
    for rec in records:
        fv = score_record(rec, threshold=args.threshold)
        rec.is_finance = fv.is_finance
        rec.finance_score = fv.score
        rec.finance_hits = fv.hits
        rec.label = assign_label(rec)

        c = per_source[rec.source]
        c["total"] += 1
        c["finance"] += rec.is_finance
        c["labelled"] += rec.label != LABEL_UNKNOWN
        c["with_image"] += any(a.local_path for a in rec.images)

    finance = [r for r in records
               if r.is_finance and r.label != LABEL_UNKNOWN
               and (not args.require_image or any(a.local_path for a in r.images))]
    for r in finance:
        per_source[r.source]["kept"] += 1

    # ---- dedup ---------------------------------------------------------
    assignment: dict[str, str] = {}
    dedup_report = None
    canonical = finance
    if not args.no_dedup and finance:
        log.info("clustering %d finance records (this is O(n^2))", len(finance))
        assignment, dedup_report = cluster_records(
            finance, data_dir=data_dir,
            image_threshold=args.image_threshold,
            text_threshold=args.text_threshold,
        )
        for r in records:
            r.raw["cluster_id"] = assignment.get(r.uid)
        canonical = pick_canonical(finance, assignment)

    # ---- write ---------------------------------------------------------
    write_records(records, interim / "curated.jsonl")
    write_records(finance, interim / "finance.jsonl")
    write_records(canonical, interim / "canonical.jsonl")

    labels = Counter(r.label for r in finance)
    report = {
        "total_raw": len(records),
        "finance_labelled": len(finance),
        "canonical": len(canonical),
        "threshold": args.threshold,
        "require_image": args.require_image,
        "labels": dict(labels),
        "per_source": {k: dict(v) for k, v in per_source.items()},
        "dedup": dedup_report.__dict__ if dedup_report else None,
    }
    write_json(report, interim / "curation_report.json")

    # ---- print ---------------------------------------------------------
    print("\n" + "=" * 78)
    print("CURATION REPORT")
    print("=" * 78)
    print(f"{'source':<28}{'raw':>7}{'finance':>9}{'labelled':>10}{'w/image':>9}{'kept':>7}")
    print("-" * 78)
    for src in sorted(per_source, key=lambda s: -per_source[s]["kept"]):
        c = per_source[src]
        print(f"{src:<28}{c['total']:>7}{c['finance']:>9}{c['labelled']:>10}"
              f"{c['with_image']:>9}{c['kept']:>7}")
    print("-" * 78)
    print(f"{'TOTAL':<28}{len(records):>7}{'':>9}{'':>10}{'':>9}{len(finance):>7}")

    print(f"\nLabel balance: {dict(labels) or '(none)'}")
    if labels:
        maj = max(labels.values()) / sum(labels.values())
        print(f"Majority-class baseline: {maj:.1%}")
        if len(labels) < 2:
            print("\n  ** ONE CLASS ONLY — cannot train a classifier on this. **")
            print("  ** See docs/CHECKLIST.md Gate 2: negative-class strategy. **")

    if dedup_report:
        print("\n" + "-" * 40)
        print("DEDUPLICATION")
        print("-" * 40)
        print(dedup_report.summary())
        if dedup_report.duplicate_rate > 0.3:
            print("\n  NOTE: >30% duplicates. Good that they were caught — this is")
            print("        exactly what would have leaked across the splits.")

    print(f"\nWrote {interim/'finance.jsonl'} and {interim/'canonical.jsonl'}")
    print("Next: python scripts/05_split.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
