#!/usr/bin/env python3
"""Calibrate the near-duplicate threshold on the real corpus.

The shipped default in `fipd.curation.dedup` is a starting point, not a
validated value — the right threshold depends on what the images actually look
like, and synthetic test images cannot tell us that.

This script estimates it from the data itself, without needing hand-labelled
duplicate pairs:

  * **Known-duplicate proxy** — records from *different* publishers whose claim
    text is highly similar are almost certainly the same viral item. Their
    image distances sample the "same creative" distribution.
  * **Known-distinct proxy** — records whose claim text shares almost nothing
    sample the "different creative" distribution.

It then prints both distributions and the threshold that best separates them,
plus what you would lose or over-merge at each candidate value.

    python scripts/calibrate_dedup.py
    python scripts/calibrate_dedup.py --sample 400

Read the overlap figure before trusting any number. If the two distributions
overlap heavily, image hashing alone is not separating this corpus and the
text signal should carry more weight.
"""

from __future__ import annotations

import argparse
import itertools
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.curation.dedup import (  # noqa: E402
    DEFAULT_IMAGE_THRESHOLD, ImageSignature, _shingles, image_distance, jaccard,
)
from fipd.utils.io import read_records  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


def histogram(values: list[float], bins: int = 20, width: int = 44) -> str:
    if not values:
        return "  (no samples)"
    lo, hi = 0.0, 1.0
    counts = [0] * bins
    for v in values:
        counts[min(int((v - lo) / (hi - lo) * bins), bins - 1)] += 1
    peak = max(counts) or 1
    out = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        lo_e, hi_e = i / bins, (i + 1) / bins
        bar = "#" * max(1, int(c / peak * width))
        out.append(f"    {lo_e:.2f}-{hi_e:.2f} |{bar} {c}")
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--in-file", default=None,
                    help="default: <data-dir>/interim/finance.jsonl")
    ap.add_argument("--sample", type=int, default=300,
                    help="max records to consider (pairwise cost is quadratic)")
    ap.add_argument("--dup-text-sim", type=float, default=0.75,
                    help="claim-similarity above which a cross-publisher pair is a likely duplicate")
    ap.add_argument("--distinct-text-sim", type=float, default=0.05,
                    help="claim-similarity below which a pair is a likely distinct item")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    log = setup()
    data_dir = Path(args.data_dir)
    in_file = Path(args.in_file) if args.in_file else data_dir / "interim" / "finance.jsonl"

    banner("Dedup threshold calibration")

    records = [r for r in read_records(in_file) if any(a.local_path for a in r.images)]
    if len(records) < 20:
        log.error("Need at least 20 records with images; found %d in %s", len(records), in_file)
        log.error("Run 01_collect.py, 02_enrich_images.py and 04_curate.py first.")
        return 1

    rng = random.Random(args.seed)
    if len(records) > args.sample:
        records = rng.sample(records, args.sample)
    log.info("calibrating on %d records", len(records))

    # --- signatures ----------------------------------------------------
    sigs: dict[str, list[ImageSignature]] = {}
    for r in records:
        got = []
        for a in r.images:
            if not a.local_path:
                continue
            s = ImageSignature.of(data_dir / a.local_path)
            if s is not None:
                got.append(s)
        if got:
            sigs[r.uid] = got
    log.info("hashed images for %d records", len(sigs))

    shingles = {r.uid: _shingles(f"{r.claim_text} {r.title}") for r in records}
    by_uid = {r.uid: r for r in records}

    dup_d: list[float] = []
    dist_d: list[float] = []

    for ra, rb in itertools.combinations([r for r in records if r.uid in sigs], 2):
        ts = jaccard(shingles[ra.uid], shingles[rb.uid])
        same_pub = (ra.source or "").split(":")[-1] == (rb.source or "").split(":")[-1]

        best = min(image_distance(sa, sb) for sa in sigs[ra.uid] for sb in sigs[rb.uid])

        if ts >= args.dup_text_sim and not same_pub:
            dup_d.append(best)
        elif ts <= args.distinct_text_sim:
            dist_d.append(best)

    print(f"\nlikely-duplicate pairs (cross-publisher, claim sim >= {args.dup_text_sim}): {len(dup_d)}")
    print(f"likely-distinct pairs  (claim sim <= {args.distinct_text_sim}):               {len(dist_d)}")

    if len(dup_d) < 5:
        print("\n  Too few duplicate pairs to calibrate. Either the corpus is small,")
        print("  or publishers genuinely are not covering the same items. Keep the")
        print(f"  default threshold ({DEFAULT_IMAGE_THRESHOLD}) and re-run when more data exists.\n")
        return 0

    print("\nimage distance — LIKELY DUPLICATES")
    print(histogram(dup_d))
    print("\nimage distance — LIKELY DISTINCT")
    print(histogram(dist_d))

    dup_d.sort()
    dist_d.sort()
    p95_dup = dup_d[int(0.95 * (len(dup_d) - 1))]
    p05_dist = dist_d[int(0.05 * (len(dist_d) - 1))] if dist_d else 1.0

    print("\n" + "-" * 66)
    print(f"{'threshold':>10}{'dup recall':>13}{'false merges':>15}{'  verdict'}")
    print("-" * 66)
    best_t, best_score = DEFAULT_IMAGE_THRESHOLD, -1.0
    for t in [x / 100 for x in range(2, 51, 2)]:
        recall = sum(d <= t for d in dup_d) / len(dup_d)
        fp = sum(d <= t for d in dist_d) / len(dist_d) if dist_d else 0.0
        # Recall is weighted higher: a missed duplicate leaks, a false merge
        # only costs diversity.
        score = 2 * recall - fp
        mark = ""
        if score > best_score:
            best_score, best_t, mark = score, t, "  <- best"
        print(f"{t:>10.2f}{recall:>12.1%}{fp:>14.1%}{mark}")
    print("-" * 66)

    overlap = max(0.0, p95_dup - p05_dist)
    print(f"\n  95th pct of duplicates : {p95_dup:.3f}")
    print(f"   5th pct of distinct    : {p05_dist:.3f}")
    print(f"  overlap                 : {overlap:.3f}", end="")
    print("   (0 = cleanly separable)" if overlap == 0 else "   <- distributions overlap")

    print(f"\n  RECOMMENDED --image-threshold {best_t:.2f}")
    print(f"  (current default is {DEFAULT_IMAGE_THRESHOLD})")
    if overlap > 0.1:
        print("\n  Heavy overlap: image hashing alone is not separating this corpus.")
        print("  Lean on the claim-text signal (raise --text-threshold weight) and")
        print("  hand-review the largest clusters before trusting the split.")
    print("\n  Then: python scripts/04_curate.py --image-threshold "
          f"{best_t:.2f}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
