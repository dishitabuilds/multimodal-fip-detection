#!/usr/bin/env python3
"""Stage 2 — filter raw fact-checks down to a labelled finance dataset.

Reads every data/raw/*.jsonl, applies the finance-relevance filter and the
verdict-to-label mapping, and writes:

    data/interim/dataset.jsonl   all records with is_finance/label filled in
    data/interim/finance.jsonl   finance-only, label != unknown
    data/interim/stats.json      counts used for the report

Also prints a yield table, which is the number that actually matters when
deciding whether the fact-check archives alone can support training.

    python scripts/build_dataset.py
    python scripts/build_dataset.py --threshold 3.0   # loosen the filter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collection.schema import FactCheckRecord, LABEL_UNKNOWN  # noqa: E402
from src.filtering.finance_filter import score_record  # noqa: E402
from src.filtering.labels import assign_label  # noqa: E402

log = logging.getLogger("dataset")


def load_raw(raw_dir: Path):
    for path in sorted(raw_dir.glob("*.jsonl")):
        with open(path, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield FactCheckRecord.from_dict(json.loads(line))
                except (json.JSONDecodeError, TypeError) as e:
                    log.warning("%s:%d unparseable (%s)", path.name, i, e)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--threshold", type=float, default=4.0, help="finance-relevance cutoff")
    ap.add_argument("--require-image", action="store_true",
                    help="drop records with no downloaded image")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    data_dir = Path(args.data_dir)
    raw_dir, interim = data_dir / "raw", data_dir / "interim"
    interim.mkdir(parents=True, exist_ok=True)

    per_source: dict[str, Counter] = defaultdict(Counter)
    label_counts: Counter = Counter()
    finance_records: list[FactCheckRecord] = []
    total = 0

    with open(interim / "dataset.jsonl", "w", encoding="utf-8") as all_fh:
        for rec in load_raw(raw_dir):
            total += 1
            src = rec.source

            fv = score_record(rec, threshold=args.threshold)
            rec.is_finance = fv.is_finance
            rec.finance_score = fv.score
            rec.finance_hits = fv.hits
            rec.label = assign_label(rec)

            per_source[src]["total"] += 1
            if rec.is_finance:
                per_source[src]["finance"] += 1
            if rec.label != LABEL_UNKNOWN:
                per_source[src]["labelled"] += 1

            has_img = any(a.local_path for a in rec.images)
            if has_img:
                per_source[src]["with_image"] += 1

            all_fh.write(rec.to_json() + "\n")

            if rec.is_finance and rec.label != LABEL_UNKNOWN:
                if args.require_image and not has_img:
                    continue
                finance_records.append(rec)
                label_counts[rec.label] += 1
                per_source[src]["kept"] += 1

    with open(interim / "finance.jsonl", "w", encoding="utf-8") as fh:
        for rec in finance_records:
            fh.write(rec.to_json() + "\n")

    stats = {
        "total_raw": total,
        "finance_labelled": len(finance_records),
        "labels": dict(label_counts),
        "threshold": args.threshold,
        "per_source": {k: dict(v) for k, v in per_source.items()},
    }
    (interim / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    # ---- report ------------------------------------------------------
    print("\n" + "=" * 74)
    print("DATASET YIELD")
    print("=" * 74)
    print(f"{'source':<26}{'raw':>8}{'finance':>10}{'labelled':>10}{'w/image':>9}{'kept':>8}")
    print("-" * 74)
    for src in sorted(per_source, key=lambda s: -per_source[s]["kept"]):
        c = per_source[src]
        print(f"{src:<26}{c['total']:>8}{c['finance']:>10}{c['labelled']:>10}"
              f"{c['with_image']:>9}{c['kept']:>8}")
    print("-" * 74)
    print(f"{'TOTAL':<26}{total:>8}{'':>10}{'':>10}{'':>9}{len(finance_records):>8}")
    print()
    print("Label balance:", dict(label_counts) or "(none)")
    if label_counts:
        maj = max(label_counts.values()) / sum(label_counts.values())
        print(f"Majority-class baseline: {maj:.1%}")
    print(f"\nWrote {interim/'finance.jsonl'} and {interim/'stats.json'}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
