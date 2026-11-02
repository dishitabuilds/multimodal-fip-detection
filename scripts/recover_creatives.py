#!/usr/bin/env python3
"""Recover article-body image candidates into a new, unreviewed corpus.

Uses the corrected body selector rather than sidebar article cards or the OG
verdict composite. Source records remain intact. Recovered images are candidates
until visually checked; no source label is silently accepted for training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.collection.article import ArticleScraper  # noqa: E402
from fipd.schema.records import LABEL_UNKNOWN  # noqa: E402
from fipd.utils.io import read_records  # noqa: E402
from fipd.utils.logging_setup import setup  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="data/interim/finance.jsonl")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--out-dir", required=True, help="new output directory")
    parser.add_argument("--uids", nargs="+", required=True)
    parser.add_argument("--cached-pages-only", action="store_true")
    parser.add_argument("--download", action="store_true", help="download/decode candidate images")
    args = parser.parse_args(argv)
    log = setup()
    output = Path(args.out_dir)
    if output.exists():
        parser.error("--out-dir must be new; existing records and reviews are preserved")
    wanted = set(args.uids)
    records = [r for r in read_records(args.records, strict=True) if r.uid in wanted]
    if {r.uid for r in records} != wanted or len(records) != len(wanted):
        parser.error("requested UIDs must each occur exactly once in the input")
    if any(not urlsplit(r.url).hostname for r in records):
        parser.error("all selected records must have an absolute source URL")
    data_dir = Path(args.data_dir)
    output.mkdir(parents=True)
    report = {"source_records": str(Path(args.records).resolve()),
              "source_sha256": hashlib.sha256(Path(args.records).read_bytes()).hexdigest(),
              "requested_records": len(records), "recovered_records": 0,
              "image_candidates": 0, "downloaded_images": 0, "results": [],
              "review_status": "pending_visual_check; no training labels accepted"}
    scrapers = {}
    with (output / "records.jsonl").open("w", encoding="utf-8") as handle:
        for original in records:
            host = urlsplit(original.url).hostname
            name = "gfc_" + host.removeprefix("www.").replace(".", "_")
            if name not in scrapers:
                scrapers[name] = ArticleScraper(name=name, base_url=f"https://{host}",
                                                sitemaps=[], out_dir=data_dir)
            scraper = scrapers[name]
            cache = scraper.http.cache_dir / (
                hashlib.sha256(original.url.encode()).hexdigest()[:24] + ".txt")
            entry = {"uid": original.uid, "url": original.url}
            if args.cached_pages_only and not cache.is_file():
                entry["status"] = "page_not_cached"
                report["results"].append(entry)
                continue
            recovered = scraper.parse_article(original.url)
            if recovered is None:
                entry["status"] = "article_unavailable"
                report["results"].append(entry)
                continue
            extraction = recovered.raw.get("image_extraction", {})
            hero = extraction.get("og_image")
            recovered.images = [a for a in recovered.images if a.url != hero]
            recovered.source, recovered.source_id = original.source, original.source_id
            recovered.label = LABEL_UNKNOWN
            recovered.raw["recovery"] = {
                "method": "article_body_v2", "source_uid": original.uid,
                "source_label": original.label, "source_verdict": original.verdict_raw,
                "previous_assets": [a.to_dict() for a in original.images],
                "page_cache_sha256": hashlib.sha256(cache.read_bytes()).hexdigest()
                if cache.is_file() else None,
                "review_status": "pending", "suggested_label": original.label,
            }
            entry["body_scope"] = extraction.get("body_scope")
            entry["candidate_urls"] = [a.url for a in recovered.images]
            report["image_candidates"] += len(recovered.images)
            if args.download:
                scraper.download_images(recovered, max_images=20)
            for asset in recovered.images:
                asset.role = "unknown"
            count = sum(bool(a.local_path) for a in recovered.images)
            entry.update(status="recovered" if recovered.images else (
                "image_downloads_failed" if entry["candidate_urls"] and args.download else "no_body_images"),
                         downloaded_images=count)
            report["downloaded_images"] += count
            report["results"].append(entry)
            if recovered.images:
                handle.write(recovered.to_json() + "\n")
                handle.flush()
                report["recovered_records"] += 1
            log.info("%s: %s (%d local images)", original.uid, entry["status"], count)
            (output / "recovery_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "recovery_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Recovered {report['recovered_records']} records / {report['downloaded_images']} local "
          f"images into {output}; visual review still required.")
    return 0 if report["recovered_records"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
