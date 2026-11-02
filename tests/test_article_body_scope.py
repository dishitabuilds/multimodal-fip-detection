"""Recover original article assets rather than the first sidebar article card."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fipd.collection.article import ArticleScraper


def parse(tmp_path, html, **kwargs):
    scraper = ArticleScraper(name="fixture", base_url="https://example.test",
                             sitemaps=[], out_dir=tmp_path, **kwargs)
    scraper.http.get_text = lambda *args, **kw: html
    return scraper.parse_article("https://example.test/claim")


def test_wordpress_body_wins_over_earlier_sidebar_article(tmp_path):
    rec = parse(tmp_path, '''<html><head><meta property="og:image" content="/verdict.jpg"></head>
        <body><aside><article><img src="/blood-donation.jpg">Unrelated sidebar</article></aside>
        <div class="post-content cf entry-content"><p>Original investment claim.</p>
        <img src="/claim-screenshot.jpg"></div></body></html>''')
    assert [a.url for a in rec.images] == ["https://example.test/verdict.jpg",
                                         "https://example.test/claim-screenshot.jpg"]
    assert rec.images[0].role == "unknown"
    assert "Original investment claim" in rec.body_text
    assert "Unrelated sidebar" not in rec.body_text
    assert rec.raw["image_extraction"]["body_scope"] == ".entry-content"


def test_article_body_wins_over_later_related_story_cards(tmp_path):
    rec = parse(tmp_path, '''<body><div class="article-body"><p>Promotion being checked.</p>
        <img src="/original.jpg"></div><article><img src="/censored-sidebar.png"></article></body>''')
    assert [a.url for a in rec.images] == ["https://example.test/original.jpg"]


def test_configured_selector_keeps_priority(tmp_path):
    rec = parse(tmp_path, '''<div class="entry-content"><img src="/other.jpg"></div>
        <section id="main-claim"><img src="/chosen.jpg"></section>''',
                article_selector="#main-claim")
    assert [a.url for a in rec.images] == ["https://example.test/chosen.jpg"]


def test_fallback_prefers_full_story_over_teaser(tmp_path):
    rec = parse(tmp_path, '''<article>Teaser<img src="/teaser.jpg"></article>
        <article>A longer actual story about a fraudulent investment promotion.
        <img src="/story.jpg"></article>''')
    assert [a.url for a in rec.images] == ["https://example.test/story.jpg"]


def test_streamed_story_body_outside_empty_article_is_found(tmp_path):
    rec = parse(tmp_path, '''<article><template id="B:0"></template></article>
        <div hidden><div class="post-article-content"><p>Investment claim evidence.</p>
        <img src="/original-promotion.jpg"></div></div>''')
    assert [a.url for a in rec.images] == ["https://example.test/original-promotion.jpg"]
    assert rec.raw["image_extraction"]["body_scope"] == ".post-article-content"
