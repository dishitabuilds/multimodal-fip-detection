"""Local mock of the real sites' response shapes, for offline testing.

Serves:
  /wp-json/wp/v2/posts        WordPress REST (with X-WP-TotalPages)
  /wp-json/wp/v2/categories
  /sitemap-daily.xml          sitemap with lastmod
  /fact-check/<slug>          article page with ClaimReview JSON-LD
  /img/<name>.jpg             small real JPEGs

Field names and header names mirror what the live endpoints actually return
(verified against altnews.in and newschecker.in), so a scraper that passes
here exercises the same code paths it will use in production.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import urlparse, parse_qs

HOST, PORT = "127.0.0.1", 8765
ROOT = f"http://{HOST}:{PORT}"

# A tiny but valid JPEG (8x8 grey), padded so it clears the min_bytes filter.
_JPEG = bytes.fromhex(
    "ffd8ffe000104a46494600010100000100010000ffdb004300ffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"
    "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffc000"
    "0b080008000801011100ffc40014000100000000000000000000000000000009ff"
    "c40014100100000000000000000000000000000000ffda0008010100003f0037ff"
    "d9"
) + b"\x00" * 6000

POSTS = [
    {
        "id": 1001,
        "date": "2026-07-14T10:00:00",
        "date_gmt": "2026-07-14T04:30:00",
        "slug": "sebi-registered-trading-app-fake",
        "link": f"{ROOT}/fact-check/sebi-registered-trading-app-fake/",
        "title": {"rendered": "Scam Alert! Viral &#8216;SEBI Registered&#8217; Trading App Promising Guaranteed Returns Is Fake"},
        "excerpt": {"rendered": "<p>A viral post claims a trading app is SEBI registered and gives assured 40% monthly profit.</p>"},
        "content": {"rendered": (
            "<p>A screenshot claiming <b>guaranteed returns</b> from an investment platform "
            "is circulating on Telegram. The creative carries a pasted SEBI logo.</p>"
            "<figure><img src='/img/creative1.jpg' alt='fake profit screenshot showing 40% return' "
            "width='1080' height='1350'/><figcaption>The viral creative</figcaption></figure>"
            "<p>BOOM found the registration number does not exist in SEBI's database.</p>"
            "<img data-src='/img/creative2.jpg' srcset='/img/small.jpg 300w, /img/creative2.jpg 1200w' "
            "alt='Telegram channel screenshot'/>"
            "<img src='/img/logo.png' alt='site logo'/>"
            "<img src='/img/fact-check-verdict-stamp.jpg' alt='verdict stamp'/>"
        )},
        "categories": [41],
        "tags": [7, 8],
        "featured_media": 55,
        "jetpack_featured_media_url": f"{ROOT}/img/hero1.jpg",
        "modified": "2026-07-14T11:00:00",
    },
    {
        "id": 1002,
        "date": "2026-06-02T09:00:00",
        "date_gmt": "2026-06-02T03:30:00",
        "slug": "deepfake-investment-video",
        "link": f"{ROOT}/fact-check/deepfake-investment-video/",
        "title": {"rendered": "Video Of Finance Minister Promoting AI Investment Platform Is A Deepfake"},
        "excerpt": {"rendered": "<p>The viral video promoting an AI trading platform is AI-generated.</p>"},
        "content": {"rendered": (
            "<p>The clip shows the minister endorsing an <i>investment platform</i> that promises "
            "to double your money in 30 days. It is a deepfake.</p>"
            "<img src='/img/creative3.jpg' alt='deepfake video thumbnail investment'/>"
        )},
        "categories": [41, 39],
        "tags": [9],
        "featured_media": 56,
        "jetpack_featured_media_url": f"{ROOT}/img/hero2.jpg",
        "modified": "2026-06-02T10:00:00",
    },
    {
        "id": 1003,
        "date": "2026-05-20T09:00:00",
        "date_gmt": "2026-05-20T03:30:00",
        "slug": "communal-video-false",
        "link": f"{ROOT}/fact-check/communal-video-false/",
        "title": {"rendered": "Old Video From Bangladesh Falsely Shared As Communal Riot In India"},
        "excerpt": {"rendered": "<p>A 2019 video is being shared with a false communal claim.</p>"},
        "content": {"rendered": (
            "<p>The video is from a 2019 protest in Dhaka and has nothing to do with any "
            "temple or mosque dispute in India. Crores of views later, the claim persists.</p>"
            "<img src='/img/creative4.jpg' alt='protest video screengrab'/>"
        )},
        "categories": [55],
        "tags": [],
        "featured_media": 57,
        "jetpack_featured_media_url": f"{ROOT}/img/hero3.jpg",
        "modified": "2026-05-20T10:00:00",
    },
]

CATEGORIES = [
    {"id": 41, "slug": "finance", "name": "Economics", "count": 11},
    {"id": 39, "slug": "news", "name": "News", "count": 934},
    {"id": 55, "slug": "religion", "name": "Religion", "count": 645},
]

SITEMAP = f"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>{ROOT}/fact-check/business/fake-sebi-advisory-telegram-32101</loc>
       <lastmod>2026-08-01T18:49:03+05:30</lastmod></url>
  <url><loc>{ROOT}/scamcheck/whatsapp-stock-tips-group-32102</loc>
       <lastmod>2026-08-02T11:20:00+05:30</lastmod></url>
  <url><loc>{ROOT}/entertainment/actor-wedding-rumour-32103</loc>
       <lastmod>2026-08-03T11:20:00+05:30</lastmod></url>
</urlset>"""

ARTICLE_TMPL = """<!doctype html><html><head>
<title>{title}</title>
<meta property="og:title" content="{title}"/>
<meta property="og:image" content="{root}/img/hero_article.jpg"/>
<meta property="og:description" content="{claim}"/>
<script type="application/ld+json">
{{"@context":"https://schema.org","@type":"ClaimReview",
 "claimReviewed":"{claim}",
 "datePublished":"2026-08-01",
 "headline":"{title}",
 "reviewRating":{{"@type":"Rating","alternateName":"{verdict}",
   "ratingValue":1,"worstRating":1,"bestRating":5}},
 "itemReviewed":{{"@type":"Claim","name":"{claim}"}}}}
</script>
<script type="application/ld+json">
{{"@context":"https://schema.org","@graph":[
 {{"@type":"NewsArticle","headline":"{title}","datePublished":"2026-08-01",
   "author":{{"@type":"Person","name":"Test Reporter"}},
   "description":"{claim}"}}]}}
</script>
</head><body>
<header><img src="{root}/img/logo.png" alt="logo"/></header>
<article class="story-content">
  <p>{body}</p>
  <figure><img src="{root}/img/article_creative.jpg" alt="viral creative"/>
  <figcaption>The viral post</figcaption></figure>
</article>
<aside class="related"><img src="{root}/img/related_thumb.jpg" alt="related story"/></aside>
</body></html>"""

ARTICLES = {
    "/fact-check/business/fake-sebi-advisory-telegram-32101": {
        "title": "Fake SEBI Advisory Circulating On Telegram Promises Guaranteed Profit",
        "claim": "A Telegram channel claims SEBI registration and guaranteed monthly returns of 30 percent",
        "verdict": "False",
        "body": "The advisory carries a forged SEBI registration number and a pasted regulator logo. "
                "It promises assured returns on a stock tips subscription.",
    },
    "/scamcheck/whatsapp-stock-tips-group-32102": {
        "title": "WhatsApp Stock Tips Group Using Doctored Profit Screenshots",
        "claim": "Screenshots show one lakh rupees turned into ten lakh through intraday trading tips",
        "verdict": "Misleading",
        "body": "The profit and loss screenshots were edited. The trading account shown does not exist.",
    },
    "/entertainment/actor-wedding-rumour-32103": {
        "title": "No, This Actor Did Not Get Married Last Week",
        "claim": "A viral photo shows the actor's wedding",
        "verdict": "False",
        "body": "The photograph is from a film shoot in 2021.",
    },
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def _send(self, code, body, ctype, extra=None):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        path = u.path

        if path == "/wp-json/wp/v2/categories":
            return self._send(200, json.dumps(CATEGORIES), "application/json")

        if path == "/wp-json/wp/v2/posts":
            posts = POSTS
            if "search" in q:
                term = q["search"][0].lower()
                posts = [p for p in posts
                         if term in json.dumps(p).lower()]
            if "categories" in q:
                cid = int(q["categories"][0])
                posts = [p for p in posts if cid in p["categories"]]
            page = int(q.get("page", ["1"])[0])
            per = int(q.get("per_page", ["50"])[0])
            total = len(posts)
            pages = max(1, (total + per - 1) // per)
            chunk = posts[(page - 1) * per: page * per]
            if page > pages:
                return self._send(400, json.dumps({"code": "rest_post_invalid_page_number"}),
                                  "application/json")
            return self._send(200, json.dumps(chunk), "application/json",
                              {"X-WP-Total": str(total), "X-WP-TotalPages": str(pages)})

        if path == "/sitemap-daily.xml":
            return self._send(200, SITEMAP, "application/xml")

        if path in ARTICLES:
            a = ARTICLES[path]
            return self._send(200, ARTICLE_TMPL.format(root=ROOT, **a), "text/html")

        if path.startswith("/img/"):
            return self._send(200, _JPEG, "image/jpeg")

        return self._send(404, "not found", "text/plain")


def serve() -> HTTPServer:
    srv = HTTPServer((HOST, PORT), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    serve()
    print(f"mock server on {ROOT}")
    threading.Event().wait()
