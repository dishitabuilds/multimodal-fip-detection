#!/usr/bin/env python3
"""Stage 03 — extract the text inside the creatives.

This is the stage that makes the project multimodal in the way the proposal
claims: the deceptive assertion lives in the image, and this is what pulls it
out so the text encoder can see it.

    # check which engines are installed on this machine
    python scripts/03_ocr.py --list-backends

    # run over everything collected so far
    python scripts/03_ocr.py --backend easyocr

    # try a different engine on the same images (cache is per-backend)
    python scripts/03_ocr.py --backend paddleocr

    # small sample first, to sanity-check quality before committing hours
    python scripts/03_ocr.py --backend easyocr --limit 20 --show

Output: data/interim/ocr/<backend>/records.jsonl with `ocr_text` populated,
plus a per-backend cache so an interrupted run never repeats work.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.enrichment.ocr import BACKENDS, get_engine, ocr_records  # noqa: E402
from fipd.utils.io import read_all, write_json, write_records  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


def list_backends() -> int:
    print("\nOCR backends:\n")
    for name in BACKENDS:
        module = "pytesseract" if name == "tesseract" else name
        try:
            installed = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            installed = False
        status = "module installed" if installed else "module NOT installed"
        if name == "tesseract":
            status += "; binary on PATH" if shutil.which("tesseract") else "; binary NOT on PATH"
        print(f"  {name:<12} {status}; uninitialized")
    print("\nOnly module/binary presence was checked. No engines were loaded or models")
    print("downloaded; model files, language data and OCR quality were not checked.")
    print("\nInstall with:  pip install -e \".[ocr]\"")
    print("Tesseract also needs the binary + `hin` traineddata.\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--backend", default="easyocr", choices=sorted(BACKENDS))
    ap.add_argument("--languages", default="en,hi")
    ap.add_argument("--in-dir", default=None, help="default: <data-dir>/raw")
    ap.add_argument("--out-dir", default=None,
                    help="separate records/report/cache directory; default: <data-dir>/interim/ocr/<backend>")
    ap.add_argument("--limit", type=int, default=None, help="only process N records")
    ap.add_argument("--gpu", action="store_true", help="EasyOCR only")
    ap.add_argument("--model-dir", default=None, help="EasyOCR model storage directory")
    ap.add_argument("--no-download", action="store_true",
                    help="EasyOCR: fail if required local model files are missing")
    ap.add_argument("--threads", type=int, default=None,
                    help="EasyOCR: cap PyTorch CPU threads during OCR")
    ap.add_argument("--include-annotated", action="store_true",
                    help="also OCR images tagged as fact-checker annotations (leaks verdicts)")
    ap.add_argument("--show", action="store_true", help="print each result as it is produced")
    ap.add_argument("--list-backends", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    if args.list_backends:
        return list_backends()
    if args.threads is not None and args.threads < 1:
        ap.error("--threads must be >= 1")
    if args.backend != "easyocr" and (
            args.model_dir or args.no_download or args.threads is not None or args.gpu):
        ap.error("--model-dir, --no-download, --threads and --gpu apply to EasyOCR only")

    log = setup(args.verbose)
    data_dir = Path(args.data_dir)
    in_dir = Path(args.in_dir) if args.in_dir else data_dir / "raw"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "interim" / "ocr" / args.backend
    if out_dir.resolve() == in_dir.resolve():
        ap.error("--out-dir must differ from --in-dir to preserve source records")
    out_dir.mkdir(parents=True, exist_ok=True)

    banner(f"Stage 03 — OCR  (backend: {args.backend})")

    records = read_all(in_dir)
    if not records:
        log.error("No records in %s — run 01_collect.py first", in_dir)
        return 1
    if args.limit:
        records = records[: args.limit]

    n_images = sum(1 for r in records for a in r.images if a.local_path)
    log.info("%d records, %d downloaded images", len(records), n_images)
    if not n_images:
        log.error("No downloaded images. Did 01_collect.py run with images enabled?")
        return 1

    options = {}
    if args.backend == "easyocr":
        options.update(gpu=args.gpu, download_enabled=not args.no_download)
        if args.model_dir:
            options["model_storage_directory"] = args.model_dir
    try:
        if args.threads is not None:
            import torch

            torch.set_num_threads(args.threads)
        engine = get_engine(args.backend,
                            languages=[l.strip() for l in args.languages.split(",") if l.strip()],
                            **options)
        engine.initialize()
    except Exception as e:
        log.error("Could not initialize %s: %s", args.backend, e)
        return 1

    stats = ocr_records(
        records, engine,
        data_dir=data_dir,
        cache_path=out_dir / "ocr_cache.json",
        skip_annotated=not args.include_annotated,
    )

    write_records(records, out_dir / "records.jsonl")

    # ---- quality report ------------------------------------------------
    with_text = [a for r in records for a in r.images if a.ocr_text]
    chars = [len(a.ocr_text) for a in with_text]
    scripts: Counter = Counter()
    for a in with_text:
        t = a.ocr_text
        dev = sum("ऀ" <= c <= "ॿ" for c in t)
        lat = sum(c.isascii() and c.isalpha() for c in t)
        if dev and lat:
            scripts["code-mixed"] += 1
        elif dev:
            scripts["devanagari"] += 1
        elif lat:
            scripts["latin"] += 1
        else:
            scripts["other"] += 1

    report = {
        "backend": args.backend,
        "languages": args.languages,
        "status": "partial_failure" if stats["errors"] else "completed",
        **stats,
        "images_with_text": len(with_text),
        "mean_chars": round(sum(chars) / len(chars), 1) if chars else 0,
        "median_chars": sorted(chars)[len(chars) // 2] if chars else 0,
        "script_mix": dict(scripts),
    }
    write_json(report, out_dir / "ocr_report.json")

    print("\n" + "=" * 62)
    print(f"OCR REPORT — {args.backend}")
    print("=" * 62)
    for k, v in report.items():
        print(f"  {k:<22} {v}")
    print()
    if stats["ocr_run"] and stats["garbage"] / max(stats["ocr_run"], 1) > 0.3:
        print("  WARNING: >30% of results look like garbage. Check the engine")
        print("           language settings, or try a different backend.\n")
    print(f"  -> {out_dir/'records.jsonl'}\n")

    if args.show:
        for a in with_text[:20]:
            print("-" * 62)
            print(a.local_path)
            print(a.ocr_text[:400])
    if stats["errors"]:
        log.error("OCR failed for %d image(s); see the cache for errors and rerun to retry",
                  stats["errors"])
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
