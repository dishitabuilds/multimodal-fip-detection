"""Google Fact Check Tools API — cross-publisher discovery layer.

Rather than crawling every Indian fact-checker separately and hoping their
themes don't change, we query Google's aggregated ClaimReview index. Every
serious fact-checker (BOOM, Factly, Newschecker, Vishvas News, India Today,
The Quint, PIB) publishes ClaimReview markup, and Google indexes all of it.

What this buys us:
  * one query surface for every publisher at once
  * the claim text and the publisher's own verdict string, already parsed
  * language filtering, so we can pull the Hindi-language checks explicitly

What it does NOT give us: the images. So this class is a *discovery* stage —
it produces URLs + labels, and `ArticleScraper.parse_article` is then pointed
at those URLs to fetch the actual creative.

Get a free key at https://console.cloud.google.com — enable "Fact Check Tools
API", create an API key, then set it in the environment:

    setx FACTCHECK_API_KEY "your-key-here"      # Windows
    export FACTCHECK_API_KEY="your-key-here"    # Linux/macOS
"""

from __future__ import annotations

import logging
import os
from typing import Iterator
from urllib.parse import urlencode, urlparse

from .base import BaseScraper
from .schema import FactCheckRecord

log = logging.getLogger(__name__)

API_ROOT = "https://factchecktools.googleapis.com/v1alpha1/claims:search"

# Publishers we care about. Anything else is dropped — the index is global and
# most of it is US politics.
INDIAN_PUBLISHERS = {
    "boomlive.in", "www.boomlive.in", "hindi.boomlive.in", "bangla.boomlive.in",
    "factly.in", "telugu.factly.in",
    "newschecker.in", "hindi.newschecker.in",
    "www.vishvasnews.com", "vishvasnews.com",
    "www.altnews.in", "altnews.in", "www.altnews.in/hindi",
    "www.indiatoday.in", "www.thequint.com",
    "pib.gov.in", "factcheck.pib.gov.in",
    "www.thehindu.com", "www.deccanherald.com",
    "digiteye.in", "www.fact-crescendo.com", "english.factcrescendo.com",
}


class GoogleFactCheckScraper(BaseScraper):
    """Query the ClaimReview index and emit records with URL + verdict."""

    name = "googlefactcheck"
    base_url = "https://factchecktools.googleapis.com"

    def __init__(self, api_key: str | None = None, restrict_to_india: bool = True, **kw) -> None:
        super().__init__(**kw)
        self.api_key = api_key or os.environ.get("FACTCHECK_API_KEY", "")
        self.restrict_to_india = restrict_to_india
        if not self.api_key:
            log.warning(
                "No FACTCHECK_API_KEY set — this source will yield nothing. "
                "See the module docstring for how to get a free key."
            )

    # ------------------------------------------------------------------
    def _search(
        self,
        query: str = "",
        language_code: str = "en",
        page_token: str = "",
        page_size: int = 100,
        publisher: str = "",
    ) -> tuple[list[dict], str]:
        """One page of results.

        The API accepts `query` (keyword) or `reviewPublisherSiteFilter`
        (everything a given publisher has ever reviewed) — at least one is
        required. Publisher mode is the higher-yield option for this project:
        it walks BOOM's whole archive rather than hoping our keyword list
        happens to match their headline wording.
        """
        params: dict[str, object] = {"key": self.api_key, "pageSize": page_size}
        if query:
            params["query"] = query
        if publisher:
            params["reviewPublisherSiteFilter"] = publisher
        if language_code:
            params["languageCode"] = language_code
        if page_token:
            params["pageToken"] = page_token

        payload, _ = self.http.get_json(f"{API_ROOT}?{urlencode(params)}", use_cache=self.use_cache)
        if not isinstance(payload, dict):
            return [], ""
        return payload.get("claims", []), payload.get("nextPageToken", "")

    @staticmethod
    def _to_records(claim: dict) -> Iterator[FactCheckRecord]:
        """One API claim can carry several publisher reviews; emit each."""
        claim_text = claim.get("text", "") or ""
        claimant = claim.get("claimant", "")
        claim_date = claim.get("claimDate", "")

        for review in claim.get("claimReview", []) or []:
            url = review.get("url", "")
            if not url:
                continue
            publisher = (review.get("publisher") or {}).get("site", "") or urlparse(url).netloc

            yield FactCheckRecord(
                source="gfc:" + publisher,
                source_id=url,
                url=url,
                title=review.get("title", ""),
                published_at=review.get("reviewDate") or claim_date or None,
                claim_text=claim_text,
                verdict_raw=review.get("textualRating", ""),
                language=review.get("languageCode", "en"),
                raw={
                    "claimant": claimant,
                    "publisher": publisher,
                    "publisher_name": (review.get("publisher") or {}).get("name", ""),
                    "claim_date": claim_date,
                },
            )

    # ------------------------------------------------------------------
    def iter_records(
        self,
        limit: int | None = None,
        queries: list[str] | None = None,
        languages: list[str] | None = None,
        publishers: list[str] | None = None,
        max_pages: int = 10,
        **_,
    ) -> Iterator[FactCheckRecord]:
        """Walk the index in two passes.

        Pass 1 (publisher sweep) pulls everything the listed publishers have
        reviewed — highest recall, and it does not depend on our keyword list
        matching their headline wording. The finance filter narrows it later.

        Pass 2 (keyword sweep) catches finance items from publishers not on
        the list, and Hindi-language items the publisher sweep may under-return.
        """
        if not self.api_key:
            return

        from ..filtering.finance_filter import DISCOVERY_QUERIES

        queries = queries or DISCOVERY_QUERIES
        languages = languages or ["en", "hi"]
        emitted = 0
        seen_urls: set[str] = set()

        def _drain(label: str, **search_kw) -> Iterator[FactCheckRecord]:
            nonlocal emitted
            token, page = "", 0
            while page < max_pages:
                claims, token = self._search(page_token=token, **search_kw)
                if not claims:
                    break
                for claim in claims:
                    for rec in self._to_records(claim):
                        if rec.url in seen_urls:
                            continue
                        host = urlparse(rec.url).netloc
                        if self.restrict_to_india and host not in INDIAN_PUBLISHERS:
                            continue
                        seen_urls.add(rec.url)
                        emitted += 1
                        yield rec
                log.info("[gfc] %s page=%d -> %d claims (%d kept overall)",
                         label, page, len(claims), emitted)
                if not token:
                    break
                page += 1

        # --- pass 1: publisher sweep ------------------------------------
        for site in publishers or []:
            for rec in _drain(f"site={site}", publisher=site, language_code=""):
                yield rec
                if limit and emitted >= limit:
                    return

        # --- pass 2: keyword sweep --------------------------------------
        for lang in languages:
            for q in queries:
                for rec in _drain(f"q={q!r} lang={lang}", query=q, language_code=lang):
                    yield rec
                    if limit and emitted >= limit:
                        return

    # ------------------------------------------------------------------
    def run(self, *a, **kw):
        """Discovery only — images are fetched later by the enrichment step."""
        kw.setdefault("with_images", False)
        return super().run(*a, **kw)
