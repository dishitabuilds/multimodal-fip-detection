"""Collection, filtering and labelling — offline against the mock server."""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))
sys.path.insert(0, str(ROOT_DIR))

from bs4 import BeautifulSoup  # noqa: E402

from tests.fixtures.mock_server import ROOT, serve  # noqa: E402

from fipd.collection.article import ArticleScraper  # noqa: E402
from fipd.collection.jsonld import (  # noqa: E402
    claimreview_from_html,
    extract_claimreview,
)
from fipd.collection.wordpress import WordPressScraper  # noqa: E402
from fipd.curation.finance_filter import score_record, score_text  # noqa: E402
from fipd.curation.labels import assign_label, map_verdict  # noqa: E402
from fipd.schema.records import FactCheckRecord  # noqa: E402
from fipd.utils.logging_setup import use_utf8_stdout  # noqa: E402

use_utf8_stdout()  # these tests print Devanagari; a cp1252 console would crash

PASS, FAIL = 0, 0


def check(cond, msg, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}   {detail}")


def main() -> int:
    serve()
    tmp = Path(tempfile.mkdtemp(prefix="mp-test-"))
    print(f"\nworkspace: {tmp}\n")

    # ---------------------------------------------------------------
    print("[1] WordPress scraper")
    wp = WordPressScraper(name="mocksite", base_url=ROOT, out_dir=tmp,
                          min_delay=0, max_delay=0)
    cats = wp.categories()
    check(cats.get(41) == "finance", "category map resolves", cats)
    check(wp.resolve_category("finance") == 41, "slug -> id lookup")

    recs = list(wp.iter_records(per_page=50))
    check(len(recs) == 3, "all posts fetched", f"got {len(recs)}")

    r = next(x for x in recs if x.source_id == "1001")
    check("SEBI Registered" in r.title and "&#8216;" not in r.title,
          "HTML entities unescaped in title", r.title)
    check("guaranteed returns" in r.body_text.lower(), "body text extracted")
    check("<p>" not in r.body_text, "tags stripped from body")

    urls = [a.url for a in r.images]
    check(not any("logo.png" in u for u in urls), "site logo excluded", urls)
    check(any("hero1.jpg" in u for u in urls), "featured image included", urls)
    check(urls[0].endswith("hero1.jpg"), "featured image first")
    check(any("creative2.jpg" in u for u in urls),
          "largest srcset variant chosen over small.jpg", urls)
    check(not any("small.jpg" in u for u in urls), "small srcset variant dropped", urls)
    check(all(u.startswith("http") for u in urls), "relative URLs resolved", urls)
    stamp = next((a for a in r.images if "verdict-stamp" in a.url), None)
    check(stamp is not None and stamp.role == "annotated",
          "verdict stamp tagged as annotated", stamp)
    cap = next((a for a in r.images if "creative1" in a.url), None)
    check(cap is not None and cap.caption == "The viral creative", "figcaption captured", cap)

    # ---------------------------------------------------------------
    print("\n[2] Image download + dedup")
    wp.download_images(r)
    check(all(a.local_path for a in r.images), "every kept image has a local path")
    check(all((tmp / a.local_path).exists() for a in r.images), "files exist on disk")
    check(all(a.sha256 for a in r.images), "hashes recorded")
    sizes = {(tmp / a.local_path).stat().st_size for a in r.images}
    check(all(s > 4096 for s in sizes), "tiny files filtered out", sizes)

    # ---------------------------------------------------------------
    print("\n[3] Sitemap + ClaimReview article scraper")
    art = ArticleScraper(
        name="mockboom", base_url=ROOT,
        sitemaps=[f"{ROOT}/sitemap-daily.xml"],
        path_filters=["/fact-check/business", "/scamcheck"],
        out_dir=tmp, min_delay=0, max_delay=0,
    )
    found = art.discover_urls()
    check(len(found) == 2, "path_filters drop the entertainment URL", [u for u, _ in found])
    check(all("lastmod" not in u for u, _ in found), "locs are clean URLs")
    check(all(lm for _, lm in found), "lastmod captured")

    a_rec = art.parse_article(*found[0])
    check(a_rec is not None, "article parsed")
    check(a_rec.verdict_raw == "False", "ClaimReview verdict extracted", a_rec.verdict_raw)
    check("SEBI registration" in a_rec.claim_text, "claimReviewed extracted", a_rec.claim_text)
    check(a_rec.author == "Test Reporter", "author from NewsArticle @graph", a_rec.author)
    check(a_rec.published_at.startswith("2026-08-01"), "datePublished used", a_rec.published_at)
    a_urls = [x.url for x in a_rec.images]
    check(any("article_creative" in u for u in a_urls), "body image found", a_urls)
    check(not any("related_thumb" in u for u in a_urls),
          "sidebar 'related' thumbnail excluded by article scoping", a_urls)
    check(not any("logo.png" in u for u in a_urls), "header logo excluded", a_urls)

    # ---------------------------------------------------------------
    print("\n[4] Finance relevance filter")
    fin = score_record(r)
    check(fin.is_finance, f"SEBI trading-app post flagged finance (score {fin.score})")
    check("sebi" in fin.hits, "sebi matched", fin.hits)

    deep = score_record(next(x for x in recs if x.source_id == "1002"))
    check(deep.is_finance, f"deepfake investment post flagged finance (score {deep.score})")

    comm = score_record(next(x for x in recs if x.source_id == "1003"))
    check(not comm.is_finance,
          f"communal-riot post rejected despite 'crores' (score {comm.score})",
          comm.negative_hits)

    hindi = score_text(title="निवेश घोटाला: पैसा डबल करने वाली स्कीम फर्जी",
                       claim="गारंटीड रिटर्न का दावा")
    check(hindi.is_finance, f"Devanagari finance text matched (score {hindi.score})", hindi.hits)

    hinglish = score_text(title="Paisa double scheme viral, SEBI registered bataya gaya",
                          claim="guaranteed profit ka dava")
    check(hinglish.is_finance, f"romanised Hinglish matched (score {hinglish.score})", hinglish.hits)

    # ---------------------------------------------------------------
    print("\n[5] Verdict -> label mapping")
    cases = [
        ("False", "fake"), ("Fake", "fake"), ("Misleading", "fake"),
        ("Partly False", "fake"), ("Misplaced Context", "fake"),
        ("Altered Photo", "fake"), ("झूठ", "fake"), ("भ्रामक", "fake"),
        ("True", "real"), ("Mostly True", "real"), ("सही", "real"),
        ("Unverified", "unknown"), ("", "unknown"), ("Research Ongoing", "unknown"),
    ]
    for raw, want in cases:
        got = map_verdict(raw)
        check(got == want, f"verdict {raw!r:22} -> {want}", f"got {got}")

    check(assign_label(FactCheckRecord(source="x", source_id="1", url="",
                                       title="Scam Alert! Viral trading app is fake")) == "fake",
          "title fallback when no verdict present")
    check(assign_label(FactCheckRecord(source="x", source_id="2", url="",
                                       title="Yes, this SEBI circular is authentic")) == "real",
          "title fallback recognises true claims")
    check(assign_label(FactCheckRecord(
        source="x", source_id="3", url="",
        title="Viral Facebook ads promoting an investment scheme are fake")) == "fake",
        "title fallback handles plural subjects ('ads ... are fake')")

    # ---------------------------------------------------------------
    print("\n[6] ClaimReview recovery")
    # The WordPress REST API carries no verdict, so it is read from the
    # article page instead. Two markup routes have to work: a plain ld+json
    # tag, and the escaped payload a Next.js app-router site streams instead.
    tag_html = """
    <html><head><script type="application/ld+json">
    {"@context":"https://schema.org","@type":"ClaimReview",
     "claimReviewed":"This trading app is SEBI approved",
     "reviewRating":{"@type":"Rating","alternateName":"False"},
     "headline":"No, this app is not SEBI approved"}
    </script></head><body>x</body></html>"""
    cr = claimreview_from_html(tag_html)
    check(cr.get("verdict") == "False", "verdict read from an ld+json tag",
          f"got {cr.get('verdict')!r}")
    check("SEBI approved" in cr.get("claim", ""), "claim read from an ld+json tag")
    check(map_verdict(cr.get("verdict", "")) == "fake", "recovered verdict maps to a label")

    # Newschecker's shape: no ld+json tag at all, the object arrives escaped
    # inside self.__next_f.push([...]) and is injected client-side.
    inner = ('{\\"@context\\":\\"https://schema.org\\",\\"@type\\":\\"ClaimReview\\",'
             '\\"claimReviewed\\":\\"Rahul Gandhi promoted an investment scheme\\",'
             '\\"reviewRating\\":{\\"alternateName\\":\\"Altered Photo/Video\\"}}')
    next_html = ('<html><body><script>self.__next_f.push([1,"8:[[\\"$\\",\\"$L17\\",null,'
                 '{\\"type\\":\\"application/ld+json\\",\\"dangerouslySetInnerHTML\\":'
                 '{\\"__html\\":\\"' + inner + '\\"}}]]"])</script></body></html>')
    check(extract_claimreview(BeautifulSoup(next_html, "html.parser")) == {},
          "the tag route finds nothing on a Next.js page — as it did in production")
    nx = claimreview_from_html(next_html)
    check(nx.get("verdict") == "Altered Photo/Video",
          "verdict recovered from the Next.js payload", f"got {nx.get('verdict')!r}")
    check("Rahul Gandhi" in nx.get("claim", ""), "claim recovered from the Next.js payload")
    check(map_verdict(nx.get("verdict", "")) == "fake",
          "'Altered Photo/Video' maps to fake")

    check(claimreview_from_html("") == {}, "empty HTML yields no verdict")
    check(claimreview_from_html("<html><body>no markup</body></html>") == {},
          "a page with no ClaimReview yields no verdict, rather than a guess")
    check(claimreview_from_html('<script>self.__next_f.push([1,"ClaimReview broken')
          == {}, "malformed payload returns empty instead of raising")

    # ---------------------------------------------------------------
    print("\n[7] Persistence + resume")
    out = wp.run(with_images=False, per_page=50)
    n1 = sum(1 for _ in open(out, encoding="utf-8"))
    out = wp.run(with_images=False, per_page=50)   # second run should add nothing
    n2 = sum(1 for _ in open(out, encoding="utf-8"))
    check(n1 == 3 and n2 == 3, "re-run is idempotent (resume works)", f"{n1} then {n2}")

    with open(out, encoding="utf-8") as fh:
        rows = [json.loads(l) for l in fh]
    check(all("uid" in row for row in rows), "uid serialised")
    check(len({row["uid"] for row in rows}) == 3, "uids unique")
    round_trip = FactCheckRecord.from_dict(rows[0])
    check(round_trip.source_id == rows[0]["source_id"], "record round-trips through JSON")
    check(isinstance(round_trip.images[0].url, str), "ImageAsset rehydrated")

    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 58)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 58 + "\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
