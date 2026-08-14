#!/usr/bin/env python3
"""Stage 1b — fetch the actual creatives for URLs found via the Google API.

The Fact Check Tools API gives us claim text and verdicts but no images. This
script takes those discovery records, visits each article page, and pulls the
images out — giving us the image half of the image+text pair.

    python scripts/enrich_images.py
    python scripts/enrich_images.py --limit 50 --finance-only
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.collection.article import ArticleScraper  # noqa: E402
from src.collection.schema import FactCheckRecord  # noqa: E402
from src.filtering.finance_filter import score_record  # noqa: E402

log = logging.getLogger("enrich")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--in-file", default=None,
                    help="default: data/raw/googlefactcheck.jsonl")
    ap.add_argument("--out-file", default=None,
                    help="default: data/raw/gfc_enriched.jsonl")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--finance-only", action="store_true",
                    help="only fetch images for records the finance filter keeps")
    ap.add_argument("--threshold", type=float, default=4.0)
    ap.add_argument("--min-delay", type=float, default=2.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(message)s")

    data_dir = Path(args.data_dir)
    in_file = Path(args.in_file) if args.in_file else data_dir / "raw" / "googlefactcheck.jsonl"
    out_file = Path(args.out_file) if args.out_file else data_dir / "raw" / "gfc_enriched.jsonl"

    if not in_file.exists():
        log.error("No discovery file at %s — run collect_factchecks.py --source googlefactcheck first",
                  in_file)
        return 1

    done: set[str] = set()
    if out_file.exists():
        with open(out_file, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["uid"])
                except (json.JSONDecodeError, KeyError):
                    continue
        log.info("resuming: %d already enriched", len(done))

    # One scraper per host so image folders and rate limiting stay per-site.
    scrapers: dict[str, ArticleScraper] = {}

    def scraper_for(url: str) -> ArticleScraper:
        host = urlparse(url).netloc
        if host not in scrapers:
            slug = host.replace("www.", "").replace(".", "_")
            scrapers[host] = ArticleScraper(
                name=f"gfc_{slug}",
                base_url=f"https://{host}",
                sitemaps=[],
                out_dir=data_dir,
                min_delay=args.min_delay,
                max_delay=args.min_delay + 1.5,
            )
        return scrapers[host]

    n_in = n_out = n_skipped = n_failed = 0

    with open(in_file, encoding="utf-8") as fin, open(out_file, "a", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                disc = FactCheckRecord.from_dict(json.loads(line))
            except (json.JSONDecodeError, TypeError):
                continue
            n_in += 1

            if disc.uid in done:
                continue
            if args.finance_only:
                fv = score_record(disc, threshold=args.threshold)
                if not fv.is_finance:
                    n_skipped += 1
                    continue

            sc = scraper_for(disc.url)
            full = sc.parse_article(disc.url)
            if full is None:
                n_failed += 1
                continue

            # Discovery metadata wins — the API's verdict is cleaner than
            # whatever we scrape off the page.
            full.source = disc.source
            full.source_id = disc.source_id
            full.verdict_raw = disc.verdict_raw or full.verdict_raw
            full.claim_text = disc.claim_text or full.claim_text
            full.language = disc.language
            full.raw.update(disc.raw)

            sc.download_images(full)
            fout.write(full.to_json() + "\n")
            fout.flush()
            done.add(full.uid)
            n_out += 1

            if n_out % 20 == 0:
                log.info("enriched %d (skipped %d, failed %d)", n_out, n_skipped, n_failed)
            if args.limit and n_out >= args.limit:
                break

    print(f"\nread {n_in} discovery records")
    print(f"enriched {n_out}, skipped {n_skipped} (non-finance), failed {n_failed}")
    print(f"-> {out_file}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
