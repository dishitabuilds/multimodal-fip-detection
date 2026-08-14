"""Generic WordPress REST API scraper.

Most Indian fact-checkers (Alt News, Newschecker, Factly, Vishvas News) run
WordPress and expose /wp-json/wp/v2/posts. That endpoint gives us clean,
paginated, structured data — far more stable than parsing their themes, which
get redesigned every year or two.

Two collection modes:
  * `search`  — run a keyword query (best for pulling finance content out of a
                mostly-political archive)
  * `browse`  — walk a category or the whole archive in date order
"""

from __future__ import annotations

import html as html_lib
import logging
import re
from typing import Iterator
from urllib.parse import urlencode

from .base import BaseScraper
from .schema import FactCheckRecord

log = logging.getLogger(__name__)


def _unescape(s: str) -> str:
    return html_lib.unescape(re.sub(r"<[^>]+>", "", s or "")).strip()


class WordPressScraper(BaseScraper):
    """Collect posts from any WordPress-backed fact-check site."""

    def __init__(self, name: str, base_url: str, **kw) -> None:
        self.name = name
        self.base_url = base_url.rstrip("/")
        super().__init__(**kw)
        self.api = f"{self.base_url}/wp-json/wp/v2"
        self._category_cache: dict[int, str] | None = None
        self._tag_cache: dict[int, str] = {}

    # ------------------------------------------------------------------
    def categories(self) -> dict[int, str]:
        if self._category_cache is not None:
            return self._category_cache
        payload, _ = self.http.get_json(f"{self.api}/categories?per_page=100")
        self._category_cache = (
            {c["id"]: c["slug"] for c in payload} if isinstance(payload, list) else {}
        )
        return self._category_cache

    def resolve_category(self, slug: str) -> int | None:
        for cid, cslug in self.categories().items():
            if cslug == slug:
                return cid
        return None

    # ------------------------------------------------------------------
    def _fetch_page(self, params: dict, page: int) -> tuple[list, int]:
        q = dict(params)
        q["page"] = page
        url = f"{self.api}/posts?{urlencode(q)}"
        payload, headers = self.http.get_json(url, use_cache=self.use_cache)
        if not isinstance(payload, list):
            return [], 0
        total_pages = int(headers.get("X-WP-TotalPages") or headers.get("x-wp-totalpages") or 0)
        return payload, total_pages

    def _to_record(self, post: dict) -> FactCheckRecord:
        content_html = (post.get("content") or {}).get("rendered", "")
        title = _unescape((post.get("title") or {}).get("rendered", ""))
        excerpt = _unescape((post.get("excerpt") or {}).get("rendered", ""))

        images = self.extract_images(content_html, page_url=post.get("link", ""))

        # The WP featured image is usually the article's hero. On fact-check
        # sites that hero is very often the viral creative itself, so keep it,
        # but put it first so ordering reflects prominence.
        feat = post.get("jetpack_featured_media_url") or ""
        if feat and not any(a.url == feat for a in images):
            from .schema import ImageAsset

            images.insert(0, ImageAsset(url=feat, role="creative", alt_text=title))

        cat_map = self.categories()
        cats = [cat_map.get(c, str(c)) for c in post.get("categories", [])]

        return FactCheckRecord(
            source=self.name,
            source_id=str(post.get("id")),
            url=post.get("link", ""),
            title=title,
            published_at=post.get("date_gmt") or post.get("date"),
            body_text=self.clean_html(content_html),
            claim_text=excerpt,
            images=images,
            categories=cats,
            tags=[str(t) for t in post.get("tags", [])],
            raw={
                "slug": post.get("slug"),
                "modified": post.get("modified"),
                "featured_media": post.get("featured_media"),
            },
        )

    # ------------------------------------------------------------------
    def iter_records(
        self,
        limit: int | None = None,
        search: str | list[str] | None = None,
        category_slug: str | None = None,
        per_page: int = 50,
        max_pages: int = 40,
        **_,
    ) -> Iterator[FactCheckRecord]:
        queries: list[dict] = []
        base = {
            "per_page": min(per_page, 100),
            "orderby": "date",
            "order": "desc",
            "_fields": (
                "id,date,date_gmt,link,slug,title,content,excerpt,categories,"
                "tags,featured_media,jetpack_featured_media_url,modified"
            ),
        }

        if search:
            terms = [search] if isinstance(search, str) else list(search)
            for t in terms:
                queries.append({**base, "search": t})
        elif category_slug:
            cid = self.resolve_category(category_slug)
            if cid is None:
                log.warning("[%s] category '%s' not found", self.name, category_slug)
                return
            queries.append({**base, "categories": cid})
        else:
            queries.append(dict(base))

        emitted = 0
        seen_ids: set[str] = set()

        for params in queries:
            label = params.get("search") or params.get("categories") or "all"
            page = 1
            while page <= max_pages:
                posts, total_pages = self._fetch_page(params, page)
                if not posts:
                    break
                for post in posts:
                    pid = str(post.get("id"))
                    if pid in seen_ids:
                        continue
                    seen_ids.add(pid)
                    yield self._to_record(post)
                    emitted += 1
                    if limit and emitted >= limit:
                        return
                log.info("[%s] query=%s page %d/%s (%d posts)",
                         self.name, label, page, total_pages or "?", len(posts))
                if total_pages and page >= total_pages:
                    break
                page += 1
