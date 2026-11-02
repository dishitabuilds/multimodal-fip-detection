"""Original-body recovery preserves source records and unreviewed status."""

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from fipd.schema.records import FactCheckRecord, ImageAsset
from fipd.utils.io import read_records

spec = importlib.util.spec_from_file_location("recovery_cli", ROOT / "scripts/recover_creatives.py")
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


@pytest.fixture
def setup_recovery(tmp_path, monkeypatch):
    original = FactCheckRecord(source="original", source_id="1", url="https://example.test/story",
                               label="fake", verdict_raw="False", title="Source title",
                               images=[ImageAsset(url="https://img.test/old.jpg")])
    source = tmp_path / "source.jsonl"
    source.write_text(original.to_json() + "\n", encoding="utf-8")
    controls = {"cached": True, "parse_calls": 0, "download_success": True}

    class Scraper:
        def __init__(self, name, base_url, sitemaps, out_dir):
            self.http = SimpleNamespace(cache_dir=Path(out_dir) / ".httpcache" / name)
            self.http.cache_dir.mkdir(parents=True, exist_ok=True)
            if controls["cached"]:
                key = hashlib.sha256(original.url.encode()).hexdigest()[:24]
                (self.http.cache_dir / f"{key}.txt").write_text("cached source HTML", encoding="utf-8")

        def parse_article(self, url):
            controls["parse_calls"] += 1
            return FactCheckRecord(source="parsed", source_id="new-slug", url=url,
                images=[ImageAsset(url="https://img.test/verdict.jpg"),
                        ImageAsset(url="https://img.test/original.jpg", role="creative")],
                raw={"image_extraction": {"og_image": "https://img.test/verdict.jpg",
                                          "body_scope": ".post-article-content"}})

        def download_images(self, rec, **kwargs):
            if not controls["download_success"]:
                rec.images = []
            else:
                for asset in rec.images:
                    asset.local_path = "images/original.jpg"

    monkeypatch.setattr(recovery, "ArticleScraper", Scraper)
    return tmp_path, original, source, controls


def args(root, rec, source):
    return ["--records", str(source), "--data-dir", str(root), "--out-dir", str(root / "output"),
            "--uids", rec.uid, "--cached-pages-only", "--download"]


def test_recovery_preserves_uid_and_source_without_approving_label(setup_recovery):
    root, rec, source, _ = setup_recovery
    before = source.read_bytes()
    assert recovery.main(args(root, rec, source)) == 0
    recovered, = list(read_records(root / "output/records.jsonl"))
    assert recovered.uid == rec.uid and recovered.label == "unknown"
    assert source.read_bytes() == before
    assert len(recovered.images) == 1
    assert recovered.images[0].url == "https://img.test/original.jpg"
    assert recovered.images[0].role == "unknown"
    assert recovered.raw["recovery"]["source_label"] == "fake"
    assert recovered.raw["recovery"]["review_status"] == "pending"


def test_cache_only_does_not_parse_uncached_pages(setup_recovery):
    root, rec, source, controls = setup_recovery
    controls["cached"] = False
    assert recovery.main(args(root, rec, source)) == 1
    assert controls["parse_calls"] == 0
    report = json.loads((root / "output/recovery_report.json").read_text())
    assert report["results"][0]["status"] == "page_not_cached"


def test_download_failure_is_not_mistaken_for_no_source_images(setup_recovery):
    root, rec, source, controls = setup_recovery
    controls["download_success"] = False
    assert recovery.main(args(root, rec, source)) == 1
    report = json.loads((root / "output/recovery_report.json").read_text())
    assert report["image_candidates"] == 1
    assert report["results"][0]["status"] == "image_downloads_failed"


def test_existing_output_and_unknown_uids_are_refused(setup_recovery):
    root, rec, source, _ = setup_recovery
    invalid = args(root, rec, source)
    invalid[invalid.index(rec.uid)] = "missing"
    with pytest.raises(SystemExit):
        recovery.main(invalid)
    assert not (root / "output").exists()
    (root / "output").mkdir()
    marker = root / "output/review.json"
    marker.write_text("saved review", encoding="utf-8")
    with pytest.raises(SystemExit):
        recovery.main(args(root, rec, source))
    assert marker.read_text() == "saved review"
