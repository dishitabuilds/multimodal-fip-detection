"""AMFI collection tests use synthetic sitemaps and images; no network access."""

from __future__ import annotations

import json
import hashlib
import sys
from pathlib import Path
from urllib.parse import quote

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.collection.legitimate import (  # noqa: E402
    MEDIA_HOST, collect_amfi, discover_candidates, download_candidate, image_url, make_record,
    select_manifest_candidates,
)
import pytest
from fipd.schema.records import FactCheckRecord  # noqa: E402


def sitemap(entries):
    nodes = []
    for page, urls in entries:
        images = "".join(f"<image:image><image:loc>{url.replace('&', '&amp;')}</image:loc>"
                         "</image:image>" for url in urls)
        nodes.append(f"<url><loc>{page}</loc>{images}</url>")
    return ('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" '
            'xmlns:image="http://www.google.com/schemas/sitemap-image/1.1">'
            + "".join(nodes) + "</urlset>")


def media(name):
    return f"https://{MEDIA_HOST}/{name}"


class FakeHTTP:
    def __init__(self, text):
        self.text = text
        self.downloads = []

    def get_text(self, url):
        return self.text

    def download(self, url, destination):
        self.downloads.append(url)
        # Large enough to pass the byte and dimension gates, entirely local.
        Image.effect_noise((180, 180), 100).convert("RGB").save(destination, format="PNG")
        return True


def test_nested_image_nodes_keep_parent_page_and_unique_url_identity():
    first, second = media("2025/start-investing.png"), media("2026/start-investing.png")
    wrapped = "https://www.mutualfundssahihai.com/_next/image?url=" + quote(first, safe="") + "&w=1000"
    text = sitemap([("https://www.mutualfundssahihai.com/en/investing", [wrapped, second]),
                    ("https://www.mutualfundssahihai.com/hi/nivesh", [first])])
    candidates = discover_candidates(text)
    assert len(candidates) == 2
    records = [make_record(c) for c in candidates]
    assert len({r.uid for r in records}) == 2  # identical filename, different URLs
    assert all(r.url.endswith(("/investing", "/nivesh")) for r in records)
    assert records[0].raw["parent_pages"] == [
        "https://www.mutualfundssahihai.com/en/investing",
        "https://www.mutualfundssahihai.com/hi/nivesh",
    ]
    assert records[0].raw["image_filename"] == "start-investing.png"


def test_no_invented_model_text_or_verified_label():
    candidate = discover_candidates(sitemap([
        ("https://www.mutualfundssahihai.com/en/campaign", [media("real-campaign.png")])]))[0]
    rec = make_record(candidate)
    assert (rec.title, rec.claim_text, rec.body_text, rec.verdict_raw) == ("", "", "", "")
    assert rec.label == "unknown" and rec.raw["suggested_label"] == "real"
    assert rec.raw["review_status"] == "pending"
    assert rec.images[0].role == "unknown"
    assert rec.images[0].caption == "" and rec.images[0].alt_text == ""
    assert not rec.is_finance


def test_exact_host_https_filter_and_icon_tokens():
    assert image_url("http://media.mutualfundssahihai.com/a.png") is None
    assert image_url("https://media.mutualfundssahihai.com.evil.org/a.png") is None
    assert image_url("https://user@media.mutualfundssahihai.com/a.png") is None
    assert image_url("https://media.mutualfundssahihai.com:444/a.png") is None
    assert image_url("https://evil.org/_next/image?url=" + quote(media("a.png"))) is None
    text = sitemap([("https://www.mutualfundssahihai.com/en/campaign", [
        media("start-investing.png"), media("amfi-logo.png"), media("icon_arrow.png"),
        "https://evil.org/a.png"]),
        ("https://evil.org/en/campaign", [media("outsider.png")])])
    assert [c.image_url for c in discover_candidates(text)] == [media("start-investing.png")]


def test_round_robin_languages_and_resumable_pilot(tmp_path):
    text = sitemap([("https://www.mutualfundssahihai.com/en/campaign", [media("en1.png"), media("en2.png")]),
                    ("https://www.mutualfundssahihai.com/hi/campaign", [media("hi1.png"), media("hi2.png")])])
    assert [c.language for c in discover_candidates(text)] == ["en", "hi", "en", "hi"]
    http = FakeHTTP(text)
    first = collect_amfi(tmp_path, limit=2, http=http)
    assert first["new_records"] == 2
    assert first["language_counts"] == {"en": 1, "hi": 1}
    second = collect_amfi(tmp_path, limit=2, http=http)
    assert second["existing_records"] == 2 and second["new_records"] == 2
    third = collect_amfi(tmp_path, limit=2, http=http)
    assert third["new_records"] == 0 and third["status"] == "ok"
    rows = [json.loads(line) for line in (tmp_path / "raw/amfi.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["uid"] for r in rows}) == 4
    assert all(r["label"] == "unknown" and r["raw"]["review_status"] == "pending" for r in rows)
    assert len(http.downloads) == 4


def test_discovery_failure_is_visible_and_does_not_create_raw_output(tmp_path):
    report = collect_amfi(tmp_path, http=FakeHTTP(None))
    assert report["status"] == "failed" and report["errors"]
    assert not (tmp_path / "raw/amfi.jsonl").exists()
    saved = json.loads((tmp_path / "processed/amfi_collection_report.json").read_text())
    assert saved["status"] == "failed"
    report = collect_amfi(tmp_path, http=FakeHTTP("not xml"))
    assert report["status"] == "failed" and "parse failure" in report["errors"][0]


def test_bad_downloads_never_publish_files_or_candidates(tmp_path):
    candidate = discover_candidates(sitemap([
        ("https://www.mutualfundssahihai.com/en/campaign", [media("a.png")])]))[0]
    rec = make_record(candidate)

    class BrokenHTTP:
        def download(self, url, destination):
            destination.write_bytes(b"not an image" * 100)
            return True

    assert not download_candidate(BrokenHTTP(), rec, tmp_path)
    assert not rec.images[0].local_path
    assert list((tmp_path / "images/amfi").iterdir()) == []


def test_existing_labelled_rows_are_preserved_and_flagged_for_review(tmp_path):
    text = sitemap([("https://www.mutualfundssahihai.com/en/campaign", [media("a.png")])])
    rec = make_record(discover_candidates(text)[0])
    rec.label = "real"
    output = tmp_path / "raw/amfi.jsonl"
    output.parent.mkdir()
    original = rec.to_json() + "\n"
    output.write_text(original, encoding="utf-8")
    report = collect_amfi(tmp_path, http=FakeHTTP(text))
    assert report["new_records"] == 0
    assert report["legacy_labelled_uids"] == [rec.uid]
    assert output.read_text() == original


def save_manifest(path, entries):
    path.write_text(json.dumps({"candidates": entries}), encoding="utf-8")
    return path


def test_priority_manifest_filters_first_batch_and_preserves_requested_parent(tmp_path):
    first, second = media("first.png"), media("second.png")
    text = sitemap([("https://www.mutualfundssahihai.com/en/a", [first, second]),
                    ("https://www.mutualfundssahihai.com/en/z", [first])])
    manifest = save_manifest(tmp_path / "priority.json", [
        {"image_url": first, "source_page_url": "https://www.mutualfundssahihai.com/en/z",
         "priority_tier": "first_batch"},
        {"image_url": second, "source_page_url": "https://www.mutualfundssahihai.com/en/a",
         "priority_tier": "boundary_probe"},
    ])
    http = FakeHTTP(text)
    report = collect_amfi(tmp_path, limit=8, http=http, candidate_file=manifest)
    assert report["new_records"] == 1 and http.downloads == [first]
    assert report["candidate_filter"]["manifest_entries"] == 2
    assert report["candidate_filter"]["filtered_other_tiers"] == 1
    assert report["candidate_filter"]["sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    row = json.loads((tmp_path / "raw/amfi.jsonl").read_text())
    assert row["url"].endswith("/en/z")
    assert row["label"] == "unknown" and row["title"] == row["claim_text"] == ""
    assert row["raw"]["review_status"] == "pending"
    assert row["raw"]["collection_candidate_manifest_sha256"] == report["candidate_filter"]["sha256"]
    both, checked = select_manifest_candidates(discover_candidates(text), manifest, "all")
    assert len(both) == 2 and not checked["errors"]


@pytest.mark.parametrize("change,reason", [
    ({"image_url": "https://evil.org/image.png"}, "untrusted"),
    ({"image_url": media("not-in-sitemap.png")}, "absent from actual sitemap"),
    ({"source_page_url": "https://evil.org/en/page"}, "untrusted"),
    ({"source_page_url": "https://www.mutualfundssahihai.com/en/unrelated"}, "does not match"),
    ({"uid": "wrong"}, "UID does not match"),
    ({"priority_tier": "invented"}, "unknown or missing priority_tier"),
])
def test_manifest_injection_is_rejected_before_any_download_or_append(tmp_path, change, reason):
    url = media("candidate.png")
    page = "https://www.mutualfundssahihai.com/en/page"
    text = sitemap([(page, [url])])
    entry = {"image_url": url, "source_page_url": page, "priority_tier": "first_batch"}
    entry.update(change)
    manifest = save_manifest(tmp_path / "priority.json", [entry])
    http = FakeHTTP(text)
    report = collect_amfi(tmp_path, http=http, candidate_file=manifest)
    assert report["status"] == "failed" and reason in " ".join(report["errors"])
    assert http.downloads == [] and not (tmp_path / "raw/amfi.jsonl").exists()


def test_duplicate_manifest_entries_and_hidden_invalid_tiers_block_entire_run(tmp_path):
    page = "https://www.mutualfundssahihai.com/en/page"
    url = media("candidate.png")
    text = sitemap([(page, [url])])
    entry = {"image_url": url, "source_page_url": page, "priority_tier": "first_batch"}
    duplicate = dict(entry, priority_tier="boundary_probe")
    manifest = save_manifest(tmp_path / "priority.json", [entry, duplicate])
    http = FakeHTTP(text)
    report = collect_amfi(tmp_path, http=http, candidate_file=manifest)
    assert report["status"] == "failed" and "duplicate image_url" in " ".join(report["errors"])
    assert http.downloads == []
