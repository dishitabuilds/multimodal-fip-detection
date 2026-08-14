#!/usr/bin/env python3
"""Stage 1 — collect fact-check articles from Indian fact-checking archives.

Examples
--------
    # everything enabled in the config, capped for a smoke test
    python scripts/collect_factchecks.py --limit 25

    # one source only, no image downloads (fast, for checking yield)
    python scripts/collect_factchecks.py --source newschecker --no-images

    # full run
    python scripts/collect_factchecks.py

Output: data/raw/<source>.jsonl, one FactCheckRecord per line, plus images
under data/images/<source>/. Runs are resumable — re-running skips records
already present in the JSONL.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collection.registry import build_scraper, load_config, run_kwargs  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to sources.yaml")
    ap.add_argument("--source", action="append", help="only run these sources (repeatable)")
    ap.add_argument("--limit", type=int, default=None, help="max records per source")
    ap.add_argument("--no-images", action="store_true", help="skip image downloads")
    ap.add_argument("--out-dir", default=None, help="override output directory")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    log = logging.getLogger("collect")

    cfg = load_config(args.config)
    defaults = cfg.get("defaults", {})
    if args.out_dir:
        defaults["out_dir"] = args.out_dir

    sources = cfg.get("sources", {})
    selected = args.source or list(sources)

    summary: list[tuple[str, int, str]] = []

    for name in selected:
        scfg = sources.get(name)
        if scfg is None:
            log.error("No such source in config: %s", name)
            continue
        if not scfg.get("enabled", True):
            log.info("Skipping disabled source: %s", name)
            continue

        log.info("=" * 62)
        log.info("Source: %s (%s)", name, scfg.get("kind"))
        log.info("=" * 62)

        scraper = build_scraper(name, scfg, defaults)
        if scraper is None:
            continue

        try:
            out = scraper.run(
                limit=args.limit,
                with_images=not args.no_images,
                **run_kwargs(scfg),
            )
        except KeyboardInterrupt:
            log.warning("Interrupted — partial results kept in data/raw/")
            return 130
        except Exception as e:  # noqa: BLE001 - one bad source must not kill the run
            log.exception("Source %s failed: %s", name, e)
            summary.append((name, 0, f"FAILED: {e}"))
            continue

        n = sum(1 for _ in open(out, encoding="utf-8")) if out.exists() else 0
        summary.append((name, n, str(out)))

    print("\n" + "=" * 62)
    print("COLLECTION SUMMARY")
    print("=" * 62)
    for name, n, where in summary:
        print(f"  {name:<20} {n:>6} records   {where}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
