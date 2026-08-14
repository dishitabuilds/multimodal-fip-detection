#!/usr/bin/env python3
"""Stage 05 — freeze reproducible, leakage-free train/val/test splits.

Splits by claim cluster rather than by record, so the same viral creative can
never appear in both training and test. Verifies that property and refuses to
write the manifest if it fails.

    python scripts/05_split.py
    python scripts/05_split.py --ratios 0.8 0.1 0.1 --seed 7
    python scripts/05_split.py --no-stratify        # ablation on stratification

Output: data/processed/split_manifest.json  — commit this file. Every reported
result should be traceable to the manifest hash it was produced under.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.curation.splits import (  # noqa: E402
    save_manifest,
    split_by_cluster,
    verify_no_leakage,
)
from fipd.utils.io import read_records, write_records  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--in-file", default=None, help="default: <data-dir>/interim/finance.jsonl")
    ap.add_argument("--ratios", nargs=3, type=float, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-stratify", action="store_true")
    ap.add_argument("--write-splits", action="store_true",
                    help="also write train/val/test JSONL files")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    log = setup(args.verbose)
    data_dir = Path(args.data_dir)
    in_file = Path(args.in_file) if args.in_file else data_dir / "interim" / "finance.jsonl"
    out_dir = data_dir / "processed"

    banner("Stage 05 — split")

    records = list(read_records(in_file))
    if not records:
        log.error("No records in %s — run 04_curate.py first", in_file)
        return 1

    # Cluster ids were written into raw by stage 04. Missing ones become
    # singletons, so this still works if dedup was skipped.
    assignment = {r.uid: (r.raw.get("cluster_id") or r.uid) for r in records}
    n_clusters = len(set(assignment.values()))
    if n_clusters == len(records):
        log.warning("every record is its own cluster — did 04_curate.py run with dedup?")

    uid_split, report = split_by_cluster(
        records, assignment,
        ratios=tuple(args.ratios), seed=args.seed,
        stratify=not args.no_stratify,
    )

    ok, problems = verify_no_leakage(uid_split, assignment, records)
    print("\n" + report.summary() + "\n")

    if not ok:
        print("LEAKAGE DETECTED — manifest not written:\n")
        for p in problems[:20]:
            print("   ", p)
        print("\nThis is a bug, not a tuning issue. Do not train until it is fixed.\n")
        return 1

    # A split with an empty val or test set passes the leakage check but is
    # useless; catch it here rather than at evaluation time.
    for name in ("train", "val", "test"):
        if report.counts.get(name, 0) == 0:
            print(f"WARNING: '{name}' split is empty. Too few clusters for these ratios.")

    path = save_manifest(uid_split, report, out_dir, assignment=assignment)

    if args.write_splits:
        for name in ("train", "val", "test"):
            subset = [r for r in records if uid_split[r.uid] == name]
            write_records(subset, out_dir / f"{name}.jsonl")

    print(f"Manifest: {path}")
    print(f"Hash:     {report.assignment_hash}   <- cite this alongside any result")
    print("\nCommit the manifest. Re-running with the same seed reproduces it exactly.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
