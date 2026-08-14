"""Base class every source scraper inherits from."""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Iterator
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from ..utils.http import PoliteSession
from .schema import FactCheckRecord, ImageAsset

log = logging.getLogger(__name__)

# Images we never want: site logos, author avatars, share icons, tracking pixels.
_IMG_BLOCKLIST = re.compile(
    r"(logo|favicon|avatar|gravatar|sprite|placeholder|whatsapp-icon|"
    r"telegram-icon|share|subscribe|banner-ad|footer|header-|/emoji/|"
    r"spacer|pixel\.gif|1x1)",
    re.I,
)

# Fact-checkers stamp their own verdict graphics onto images. Those are
# annotations, not the original creative, and must not leak into training.
_ANNOTATION_HINTS = re.compile(
    r"(fact-?check(ed)?|verdict|false-?stamp|boom-?verdict|misleading-?stamp|"
    r"claim-?check|watermark|newschecker-?stamp|alt-?news-?stamp)",
    re.I,
)


class BaseScraper(ABC):
    """Common plumbing: image extraction, HTML cleanup, JSONL persistence."""

    #: short machine name, also used as the folder name under data/
    name: str = "base"
    #: site root, used to resolve protocol-relative and root-relative URLs
    base_url: str = ""

    def __init__(
        self,
        out_dir: str | Path = "data",
        min_delay: float = 1.5,
        max_delay: float = 3.0,
        use_cache: bool = True,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.raw_dir = self.out_dir / "raw"
        self.image_dir = self.out_dir / "images" / self.name
        self.cache_dir = self.out_dir / ".httpcache" / self.name
        for d in (self.raw_dir, self.image_dir, self.cache_dir):
            d.mkdir(parents=True, exist_ok=True)

        self.use_cache = use_cache
        self.http = PoliteSession(
            min_delay=min_delay, max_delay=max_delay, cache_dir=self.cache_dir
        )

    # ------------------------------------------------------------------
    # Subclasses implement this
    # ------------------------------------------------------------------
    @abstractmethod
    def iter_records(self, limit: int | None = None, **kw) -> Iterator[FactCheckRecord]:
        """Yield normalised records from this source."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------
    @staticmethod
    def clean_html(html: str) -> str:
        """Strip tags, scripts and boilerplate; return readable plain text."""
        if not html:
            return ""
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "iframe", "form"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r"[ \t]{2,}", " ", text)
        return text.strip()

    def extract_images(self, html: str, page_url: str = "") -> list[ImageAsset]:
        """Pull candidate images out of an article body.

        Handles lazy-loading (`data-src`, `data-lazy-src`) and `srcset`,
        picking the largest declared variant so OCR has something legible.
        """
        if not html:
            return []
        soup = BeautifulSoup(html, "html.parser")
        assets: list[ImageAsset] = []
        seen: set[str] = set()

        for img in soup.find_all("img"):
            url = (
                img.get("data-src")
                or img.get("data-lazy-src")
                or img.get("data-original")
                or img.get("src")
                or ""
            ).strip()

            # Prefer the widest candidate in srcset when present.
            srcset = img.get("data-srcset") or img.get("srcset") or ""
            if srcset:
                best, best_w = url, 0
                for part in srcset.split(","):
                    bits = part.strip().split()
                    if not bits:
                        continue
                    cand = bits[0]
                    width = 0
                    if len(bits) > 1 and bits[1].endswith("w"):
                        try:
                            width = int(bits[1][:-1])
                        except ValueError:
                            width = 0
                    if width >= best_w:
                        best, best_w = cand, width
                if best:
                    url = best

            if not url or url.startswith("data:"):
                continue
            url = urljoin(page_url or self.base_url, url)
            if url in seen or _IMG_BLOCKLIST.search(url):
                continue
            seen.add(url)

            alt = (img.get("alt") or "").strip()
            caption = ""
            fig = img.find_parent("figure")
            if fig:
                cap = fig.find("figcaption")
                if cap:
                    caption = cap.get_text(" ", strip=True)

            role = "annotated" if _ANNOTATION_HINTS.search(f"{url} {alt} {caption}") else "creative"

            def _int(v):
                try:
                    return int(str(v).replace("px", ""))
                except (TypeError, ValueError):
                    return None

            assets.append(
                ImageAsset(
                    url=url,
                    role=role,
                    alt_text=alt,
                    caption=caption,
                    width=_int(img.get("width")),
                    height=_int(img.get("height")),
                )
            )
        return assets

    def download_images(
        self, record: FactCheckRecord, max_images: int = 8, min_bytes: int = 4096
    ) -> None:
        """Fetch each image to disk and record its sha256.

        Tiny files (spacers, broken responses) are dropped, and files that
        hash-collide with something already downloaded are de-duplicated —
        fact-checkers reuse the same stock art across dozens of articles.
        """
        kept: list[ImageAsset] = []
        for asset in record.images[:max_images]:
            ext = Path(urlparse(asset.url).path).suffix.lower()
            if ext not in {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp"}:
                ext = ".jpg"
            key = hashlib.sha256(asset.url.encode()).hexdigest()[:20]
            dest = self.image_dir / f"{record.uid}_{key}{ext}"

            if not self.http.download(asset.url, dest):
                continue
            if dest.stat().st_size < min_bytes:
                dest.unlink(missing_ok=True)
                continue

            asset.local_path = str(dest.relative_to(self.out_dir))
            asset.sha256 = hashlib.sha256(dest.read_bytes()).hexdigest()
            kept.append(asset)

        record.images = kept

    # ------------------------------------------------------------------
    def run(
        self,
        limit: int | None = None,
        with_images: bool = True,
        out_file: str | Path | None = None,
        **kw,
    ) -> Path:
        """Collect records and append them to a JSONL file. Resumable."""
        out_file = Path(out_file) if out_file else self.raw_dir / f"{self.name}.jsonl"
        out_file.parent.mkdir(parents=True, exist_ok=True)

        already: set[str] = set()
        if out_file.exists():
            with open(out_file, encoding="utf-8") as fh:
                for line in fh:
                    try:
                        already.add(json.loads(line)["uid"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            log.info("[%s] resuming, %d records already on disk", self.name, len(already))

        n_new = 0
        with open(out_file, "a", encoding="utf-8") as fh:
            for rec in self.iter_records(limit=limit, **kw):
                if rec.uid in already:
                    continue
                if with_images:
                    self.download_images(rec)
                fh.write(rec.to_json() + "\n")
                fh.flush()
                already.add(rec.uid)
                n_new += 1
                if n_new % 25 == 0:
                    log.info("[%s] %d new records", self.name, n_new)

        log.info("[%s] done: %d new, %d total -> %s", self.name, n_new, len(already), out_file)
        return out_file
