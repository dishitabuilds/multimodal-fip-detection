#!/usr/bin/env python3
"""Stage 01b — collect legitimate financial promotion creatives (Negative Class / Gate 2).

Fact-checking archives only publish debunks (100% fake class).
To train a binary classifier, this script harvests genuine, authorised financial
promotion creatives and investor education banners from:
  1. AMFI (Association of Mutual Funds in India) — "Mutual Funds Sahi Hai" campaign
  2. SEBI (Securities and Exchange Board of India) — Investor Education portal

These match the EXACT visual genre of scam creatives (promotional banners,
finance claims, Hindi/English slogans, celebrity endorsements like Rohit Sharma)
differing only in being legitimate and authorised.

Examples
--------
    # Collect 100 legitimate creatives with images
    python scripts/01b_collect_legitimate.py --limit 100

    # Collect without downloading images (metadata only)
    python scripts/01b_collect_legitimate.py --no-images

Output:
    data/raw/amfi.jsonl
    data/images/amfi/<uid>_<hash>.<ext>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from pathlib import Path
from urllib.parse import unquote, urlparse

import requests
from bs4 import BeautifulSoup
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.schema.records import FactCheckRecord, ImageAsset, LABEL_REAL  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


# Fund houses and UI icons to skip so we only keep real campaign posters/infographics
_SKIP_PATTERNS = re.compile(
    r"(logo|icon|favicon|sprite|bandhan|axis|dsp|franklin|hdfc|icici|"
    r"kotak|nippon|sbi-|tata-|uti-|mirae|hsbc|invesco|canara|sundaram|"
    r"pgim|motilal|whiteoak|quant-|groww-|navi-|zerodha-|edelweiss|"
    r"360-one|union-|baroda|trust-|samco|ppfas|iti-|bank-|calculator|"
    r"footer|header|menu|arrow|star|rating)",
    re.I,
)


def _clean_title(slug: str) -> str:
    """Turn an image filename/slug into a readable promotional title."""
    slug = re.sub(r"\.(webp|png|jpg|jpeg|gif)$", "", slug, flags=re.I)
    slug = re.sub(r"_\d+x\d+.*$", "", slug)  # remove resolution suffixes like _700x700
    cleaned = slug.replace("_", " ").replace("-", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.title()


def _fetch_amfi_creatives(session: requests.Session) -> list[tuple[str, str, str]]:
    """Return list of (image_url, title, language) from AMFI campaign sitemap."""
    sitemap_url = "https://www.mutualfundssahihai.com/sitemap-0.xml"
    try:
        r = session.get(sitemap_url, timeout=15)
        r.raise_for_status()
    except Exception as e:
        logging.error("Failed to fetch AMFI sitemap: %s", e)
        return []

    soup = BeautifulSoup(r.text, "xml")
    locs = [loc.text for loc in soup.find_all("loc")]

    items = []
    seen_urls = set()

    for loc in locs:
        # Sitemap contains entries like:
        # /_next/image?url=https%3A%2F%2Fmedia.mutualfundssahihai.com%2F...
        m = re.search(r"url=([^&]+)", loc)
        img_url = unquote(m.group(1)) if m else loc

        if "media.mutualfundssahihai.com" not in img_url:
            continue
        if img_url in seen_urls:
            continue
        seen_urls.add(img_url)

        filename = img_url.split("/")[-1]
        if _SKIP_PATTERNS.search(filename):
            continue

        title = _clean_title(filename)
        if len(title) < 5 or title.lower() in {"explore more", "learn", "invest", "contact1", "empower"}:
            continue

        lang = "hi" if "hindi" in filename.lower() else "en"
        items.append((img_url, title, lang))

    return items


def _download_image(session: requests.Session, url: str, dest: Path) -> tuple[bool, int, int]:
    """Download image to dest and return (success, width, height)."""
    try:
        r = session.get(url, timeout=20)
        if r.status_code != 200 or len(r.content) < 4096:
            return False, 0, 0
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(r.content)

        # Inspect dimensions with PIL
        with Image.open(dest) as img:
            w, h = img.size
            # Filter out tiny badge icons
            if w < 120 or h < 120:
                dest.unlink(missing_ok=True)
                return False, 0, 0
            return True, w, h
    except Exception as e:
        logging.debug("Download failed for %s: %s", url, e)
        dest.unlink(missing_ok=True)
        return False, 0, 0


def collect_amfi(
    out_dir: Path,
    limit: int = 100,
    download_images: bool = True,
    log: logging.Logger = logging.getLogger(__name__),
) -> int:
    """Harvest legitimate records from AMFI."""
    raw_dir = out_dir / "raw"
    images_dir = out_dir / "images" / "amfi"
    raw_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    out_file = raw_dir / "amfi.jsonl"
    already_uids = set()
    if out_file.exists():
        for line in out_file.read_text(encoding="utf-8").splitlines():
            try:
                already_uids.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                pass
        log.info("[amfi] Resuming, %d records already on disk", len(already_uids))

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko)"
    })

    log.info("[amfi] Discovering legitimate campaign creatives from AMFI...")
    candidates = _fetch_amfi_creatives(session)
    log.info("[amfi] Found %d unique campaign candidates", len(candidates))

    written = 0
    with open(out_file, "a", encoding="utf-8") as fh:
        for img_url, title, lang in candidates:
            if limit and written >= limit:
                break

            slug = re.sub(r"[^\w\s-]", "", title.lower())
            slug = re.sub(r"\s+", "-", slug).strip("-")
            source_id = f"amfi-{slug}"
            uid = hashlib.sha1(f"amfi:{source_id}".encode()).hexdigest()[:16]

            if uid in already_uids:
                continue

            asset = ImageAsset(
                url=img_url,
                role="creative",
                alt_text=title,
                caption=f"Official AMFI Mutual Funds Sahi Hai promotional creative: {title}",
            )

            if download_images:
                ext = Path(urlparse(img_url).path).suffix.lower() or ".webp"
                key = hashlib.sha256(img_url.encode()).hexdigest()[:20]
                dest = images_dir / f"{uid}_{key}{ext}"

                ok, w, h = _download_image(session, img_url, dest)
                if not ok:
                    continue

                asset.local_path = str(dest.relative_to(out_dir))
                asset.width = w
                asset.height = h
                asset.sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
                time.sleep(0.3)  # polite crawl delay

            rec = FactCheckRecord(
                source="amfi",
                source_id=source_id,
                url=f"https://www.mutualfundssahihai.com/en/search?q={requests.utils.quote(title)}",
                title=f"AMFI Investor Education: {title}",
                claim_text=f"Mutual Funds Sahi Hai — {title}. Investing in mutual funds is subject to market risk, read scheme documents carefully.",
                body_text=(
                    f"Official investor education campaign material by the Association of Mutual Funds in India (AMFI). "
                    f"Campaign topic: {title}. Legitimate, registered mutual fund awareness initiative under SEBI guidelines."
                ),
                verdict_raw="Genuine",
                label=LABEL_REAL,
                is_finance=True,
                finance_score=25.0,
                finance_hits=["amfi", "mutual fund", "investment", "sebi"],
                images=[asset],
                language=lang,
                categories=["mutual funds", "investor education", "legitimate promotion"],
                raw={"campaign": "Mutual Funds Sahi Hai", "origin": img_url},
            )

            fh.write(rec.to_json() + "\n")
            already_uids.add(uid)
            written += 1
            if written % 10 == 0:
                log.info("[amfi] Harvested %d / %d legitimate items", written, limit)

    log.info("[amfi] Finished. Total newly written: %d records -> %s", written, out_file)
    return written


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data", help="Root data directory (default: data)")
    ap.add_argument("--limit", type=int, default=100, help="Number of legitimate records to harvest (default: 100)")
    ap.add_argument("--no-images", action="store_true", help="Skip downloading image files")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    log = setup(args.verbose)
    banner(f"Stage 01b — Collect Legitimate Promos (AMFI) (target: {args.limit})")

    out_dir = Path(args.data_dir)
    n = collect_amfi(out_dir, limit=args.limit, download_images=not args.no_images, log=log)
    log.info("Done. %d legitimate records ready for curation.", n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
