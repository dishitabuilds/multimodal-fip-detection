"""Sitemap-driven article scraper for non-WordPress fact-check sites.

Used for BOOM Live (Quintype/Next.js) and as a fallback for any site whose
REST API is behind Cloudflare. Two stages:

  1. discover article URLs from the site's XML sitemaps, filtered by path
  2. fetch each article, extract structured data (schema.org ClaimReview when
     present, which is the cleanest source of claim + verdict) and images

ClaimReview is what Google reads to build its fact-check panels, so almost
every serious fact-checker emits it. When it is missing we fall back to
Open Graph tags and the article body.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Iterator
from xml.etree import ElementTree as ET

from bs4 import BeautifulSoup

from .base import BaseScraper
from ..schema.records import FactCheckRecord

log = logging.getLogger(__name__)

_SITEMAP_NS = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}


class ArticleScraper(BaseScraper):
    """Generic sitemap + JSON-LD article collector."""

    def __init__(
        self,
        name: str,
        base_url: str,
        sitemaps: list[str],
        path_filters: list[str] | None = None,
        article_selector: str | None = None,
        **kw,
    ) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        super().__init__(**kw)
        self.sitemaps = sitemaps
        # Only keep URLs whose path contains one of these fragments.
        self.path_filters = path_filters or []
        # CSS selector for the article body; falls back to <article>/og heuristics.
        self.article_selector = article_selector

    # ------------------------------------------------------------------
    # Stage 1: URL discovery
    # ------------------------------------------------------------------
    def _parse_sitemap(self, url: str, depth: int = 0) -> list[tuple[str, str]]:
        """Return [(loc, lastmod)]. Recurses one level into sitemap indexes."""
        if depth > 2:
            return []
        xml = self.http.get_text(url, use_cache=self.use_cache)
        if not xml:
            return []
        try:
            root = ET.fromstring(xml.encode("utf-8"))
        except ET.ParseError as e:
            log.warning("[%s] bad sitemap XML at %s: %s", self.name, url, e)
            return []

        tag = root.tag.split("}")[-1]
        out: list[tuple[str, str]] = []

        if tag == "sitemapindex":
            for sm in root.findall("sm:sitemap", _SITEMAP_NS):
                loc = sm.findtext("sm:loc", default="", namespaces=_SITEMAP_NS)
                if loc:
                    out.extend(self._parse_sitemap(loc, depth + 1))
            return out

        for u in root.findall("sm:url", _SITEMAP_NS):
            loc = u.findtext("sm:loc", default="", namespaces=_SITEMAP_NS)
            lastmod = u.findtext("sm:lastmod", default="", namespaces=_SITEMAP_NS)
            if loc:
                out.append((loc, lastmod))
        return out

    def discover_urls(self) -> list[tuple[str, str]]:
        seen: set[str] = set()
        urls: list[tuple[str, str]] = []
        for sm in self.sitemaps:
            for loc, lastmod in self._parse_sitemap(sm):
                if loc in seen:
                    continue
                if self.path_filters and not any(f in loc for f in self.path_filters):
                    continue
                seen.add(loc)
                urls.append((loc, lastmod))
        log.info("[%s] discovered %d article URLs", self.name, len(urls))
        return urls

    # ------------------------------------------------------------------
    # Stage 2: article parsing
    # ------------------------------------------------------------------
    @staticmethod
    def _iter_jsonld(soup: BeautifulSoup) -> Iterator[dict]:
        for tag in soup.find_all("script", type="application/ld+json"):
            try:
                data = json.loads(tag.string or tag.get_text() or "{}")
            except (json.JSONDecodeError, TypeError):
                continue
            items = data if isinstance(data, list) else [data]
            for item in items:
                if not isinstance(item, dict):
                    continue
                yield item
                for sub in item.get("@graph", []) or []:
                    if isinstance(sub, dict):
                        yield sub

    def _extract_claimreview(self, soup: BeautifulSoup) -> dict:
        """Return {claim, verdict, published, title} from ClaimReview if present."""
        for item in self._iter_jsonld(soup):
            if item.get("@type") not in ("ClaimReview", ["ClaimReview"]):
                continue
            rating = item.get("reviewRating") or {}
            claim = item.get("claimReviewed") or ""
            appearance = item.get("itemReviewed") or {}
            if not claim and isinstance(appearance, dict):
                claim = appearance.get("name", "")
            return {
                "claim": claim if isinstance(claim, str) else "",
                "verdict": (rating.get("alternateName") or rating.get("name") or ""),
                "rating_value": rating.get("ratingValue"),
                "worst_rating": rating.get("worstRating"),
                "best_rating": rating.get("bestRating"),
                "published": item.get("datePublished", ""),
                "title": (item.get("headline") or ""),
            }
        return {}

    def _extract_article_meta(self, soup: BeautifulSoup) -> dict:
        for item in self._iter_jsonld(soup):
            t = item.get("@type")
            types = t if isinstance(t, list) else [t]
            if not any(x in ("NewsArticle", "Article", "BlogPosting") for x in types):
                continue
            author = item.get("author") or {}
            if isinstance(author, list):
                author = author[0] if author else {}
            return {
                "title": item.get("headline", ""),
                "published": item.get("datePublished", ""),
                "author": author.get("name", "") if isinstance(author, dict) else str(author),
                "description": item.get("description", ""),
            }
        return {}

    def parse_article(self, url: str, lastmod: str = "") -> FactCheckRecord | None:
        html = self.http.get_text(url, use_cache=self.use_cache)
        if not html:
            return None
        soup = BeautifulSoup(html, "html.parser")

        cr = self._extract_claimreview(soup)
        meta = self._extract_article_meta(soup)

        def og(prop: str) -> str:
            tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
            return (tag.get("content") or "").strip() if tag else ""

        title = cr.get("title") or meta.get("title") or og("og:title") or (
            soup.title.get_text(strip=True) if soup.title else ""
        )

        # Narrow to the article body before pulling images, so we don't hoover
        # up "related stories" thumbnails from the sidebar.
        body_node = None
        if self.article_selector:
            body_node = soup.select_one(self.article_selector)
        if body_node is None:
            body_node = soup.find("article") or soup.find(
                attrs={"class": re.compile(r"(story|article|post)[-_]?(body|content|detail)", re.I)}
            )
        body_html = str(body_node) if body_node else str(soup.find("body") or soup)

        images = self.extract_images(body_html, page_url=url)
        hero = og("og:image")
        if hero and not any(a.url == hero for a in images):
            from ..schema.records import ImageAsset

            images.insert(0, ImageAsset(url=hero, role="creative", alt_text=title))

        slug = url.rstrip("/").split("/")[-1]

        return FactCheckRecord(
            source=self.name,
            source_id=slug,
            url=url,
            title=title,
            published_at=cr.get("published") or meta.get("published") or lastmod or None,
            author=meta.get("author", ""),
            body_text=self.clean_html(body_html),
            claim_text=cr.get("claim") or meta.get("description") or og("og:description"),
            verdict_raw=cr.get("verdict", ""),
            images=images,
            categories=[p for p in url.replace(self.base_url, "").split("/") if p][:-1],
            raw={"lastmod": lastmod, "claimreview": cr or None},
        )

    # ------------------------------------------------------------------
    def iter_records(self, limit: int | None = None, **_) -> Iterator[FactCheckRecord]:
        urls = self.discover_urls()
        n = 0
        for url, lastmod in urls:
            rec = self.parse_article(url, lastmod)
            if rec is None:
                continue
            yield rec
            n += 1
            if limit and n >= limit:
                return
