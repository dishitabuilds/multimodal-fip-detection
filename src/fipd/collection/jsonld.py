"""schema.org JSON-LD extraction, shared by every scraper.

ClaimReview is the cleanest source of claim-plus-verdict there is: it is what
Google reads to build its fact-check panels, so serious fact-checkers emit it
and keep it accurate. It lives in the article HTML.

That last sentence is why this module is shared rather than living inside the
sitemap scraper. The WordPress REST API returns clean structured post data and
**no verdict at all** — measured on a real run, 217 of 269 finance-relevant
records (81%) came back with an empty `verdict_raw` and were dropped as
unlabelled, while the verdict sat in ClaimReview markup on the article page the
REST route never fetches. Recovering it costs one extra request per record and
is the difference between a few dozen trainable items and a few hundred.
"""

from __future__ import annotations

import json
import logging
from typing import Iterator

from bs4 import BeautifulSoup

log = logging.getLogger(__name__)

#: Marker for ClaimReview inside a Next.js streamed payload. Newschecker (and
#: any other Next.js app-router site) does not emit a real
#: <script type="application/ld+json"> tag — the markup is injected client-side
#: from an escaped JSON string inside self.__next_f.push([...]), so BeautifulSoup
#: finds nothing and the verdict looks absent when it is right there.
_NEXTJS_MARKER = '"@type":"ClaimReview"'


def iter_jsonld(soup: BeautifulSoup) -> Iterator[dict]:
    """Yield every JSON-LD object on the page, including @graph members."""
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


def _as_result(item: dict) -> dict:
    """Normalise one ClaimReview object into the fields collection needs."""
    rating = item.get("reviewRating") or {}
    claim = item.get("claimReviewed") or ""
    if not claim:
        reviewed = item.get("itemReviewed") or {}
        if isinstance(reviewed, dict):
            claim = reviewed.get("name", "")

    return {
        "claim": claim if isinstance(claim, str) else "",
        "verdict": (rating.get("alternateName") or rating.get("name") or ""),
        "rating_value": rating.get("ratingValue"),
        "worst_rating": rating.get("worstRating"),
        "best_rating": rating.get("bestRating"),
        "published": item.get("datePublished", ""),
        "title": (item.get("headline") or ""),
    }


def extract_claimreview(soup: BeautifulSoup) -> dict:
    """Return {claim, verdict, ...} from ClaimReview markup, or {} if absent."""
    for item in iter_jsonld(soup):
        types = item.get("@type")
        types = types if isinstance(types, list) else [types]
        if "ClaimReview" in types:
            return _as_result(item)
    return {}


def extract_claimreview_nextjs(html: str) -> dict:
    """Recover ClaimReview from a Next.js streamed payload.

    The object is present as an escaped JSON string inside a `<script>` that
    pushes onto `self.__next_f`, so there is no ld+json tag for BeautifulSoup
    to find. Unescaping the document and decoding from the object's opening
    brace gets it back.

    Measured on Newschecker: 0 records recovered through the tag route, because
    every one of its verdicts is here instead.
    """
    if "ClaimReview" not in html:
        return {}

    # Collapse the escaping the payload was serialised with. Longest first, so
    # \\" does not get half-converted by the \" rule.
    unescaped = html.replace('\\\\"', '"').replace('\\"', '"').replace("\\\\", "\\")

    decoder = json.JSONDecoder()
    idx = unescaped.find(_NEXTJS_MARKER)
    while idx != -1:
        start = unescaped.rfind("{", 0, idx)
        if start != -1:
            try:
                obj, _ = decoder.raw_decode(unescaped[start:])
            except json.JSONDecodeError:
                obj = None
            if isinstance(obj, dict):
                types = obj.get("@type")
                types = types if isinstance(types, list) else [types]
                if "ClaimReview" in types:
                    return _as_result(obj)
        idx = unescaped.find(_NEXTJS_MARKER, idx + 1)
    return {}


def claimreview_from_html(html: str) -> dict:
    """ClaimReview from raw HTML, trying both the tag and the Next.js route."""
    if not html:
        return {}
    found = extract_claimreview(BeautifulSoup(html, "html.parser"))
    if found.get("verdict"):
        return found
    return extract_claimreview_nextjs(html) or found
