"""Downloaded image bytes, rather than HTML layout, determine pixel geometry."""

from __future__ import annotations

import hashlib
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.collection.base import BaseScraper  # noqa: E402
from fipd.datasets.examples import pick_image  # noqa: E402
from fipd.schema.records import FactCheckRecord, ImageAsset  # noqa: E402


class OfflineScraper(BaseScraper):
    name = "offline"

    def iter_records(self, limit=None, **kwargs):
        return iter(())


def image_bytes(size, format="PNG"):
    output = io.BytesIO()
    Image.new("RGB", size, color=(70, 80, 90)).save(output, format=format)
    return output.getvalue()


class ImageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.scraper = OfflineScraper(out_dir=self.root, min_delay=0, max_delay=0)
        self.payloads = {}

        def download(url, destination):
            destination.write_bytes(self.payloads[url])
            return True

        self.scraper.http.download = Mock(side_effect=download)

    def record(self, assets):
        return FactCheckRecord(source="offline", source_id="1", url="https://example.com/post",
                               images=assets)

    def test_actual_geometry_overrides_absent_and_stale_html_sizes(self):
        hero = ImageAsset(url="https://example.com/hero.png", role="creative",
                          alt_text="Original claim", caption="Source caption")
        sidebar = ImageAsset(url="https://example.com/sidebar.png", role="creative",
                             width=811, height=431)
        self.payloads[hero.url] = image_bytes((2560, 1283))
        self.payloads[sidebar.url] = image_bytes((1000, 667))
        record = self.record([sidebar, hero])
        self.scraper.download_images(record, min_bytes=0)

        self.assertEqual((hero.width, hero.height), (2560, 1283))
        self.assertEqual((sidebar.width, sidebar.height), (1000, 667))
        self.assertEqual(pick_image(record), hero.local_path)
        self.assertEqual(hero.url, "https://example.com/hero.png")
        self.assertEqual(hero.role, "creative")
        self.assertEqual(hero.alt_text, "Original claim")
        self.assertEqual(hero.caption, "Source caption")
        self.assertEqual(hero.sha256, hashlib.sha256(self.payloads[hero.url]).hexdigest())
        self.assertEqual((self.root / hero.local_path).read_bytes(), self.payloads[hero.url])

    def test_html_disguised_as_jpeg_is_excluded_and_deleted(self):
        asset = ImageAsset(url="https://example.com/broken.jpg", role="creative",
                           width=4000, height=3000)
        self.payloads[asset.url] = b"<html>Service unavailable</html>" * 300
        record = self.record([asset])
        with self.assertLogs("fipd.collection.base", level="WARNING"):
            self.scraper.download_images(record)
        self.assertEqual(record.images, [])
        self.assertEqual(list(self.scraper.image_dir.iterdir()), [])
        self.assertIsNone(asset.local_path)
        self.assertIsNone(asset.sha256)

    def test_truncated_image_cannot_pass_header_only_validation(self):
        asset = ImageAsset(url="https://example.com/truncated.jpg")
        jpeg = image_bytes((1000, 667), format="JPEG")
        self.payloads[asset.url] = jpeg[:len(jpeg) // 2]
        record = self.record([asset])
        with self.assertLogs("fipd.collection.base", level="WARNING"):
            self.scraper.download_images(record, min_bytes=0)
        self.assertEqual(record.images, [])
        self.assertEqual(list(self.scraper.image_dir.iterdir()), [])

    def test_annotation_role_is_preserved_despite_readable_pixels(self):
        asset = ImageAsset(url="https://example.com/annotation.png", role="annotated",
                           width=1, height=1)
        self.payloads[asset.url] = image_bytes((400, 300))
        record = self.record([asset])
        self.scraper.download_images(record, min_bytes=0)
        self.assertEqual((asset.width, asset.height), (400, 300))
        self.assertEqual(asset.role, "annotated")
        self.assertIsNone(pick_image(record))

    def test_corrupt_png_checksum_is_excluded(self):
        asset = ImageAsset(url="https://example.com/corrupt.png")
        payload = bytearray(image_bytes((400, 300)))
        payload[payload.index(b"IDAT") + 4] ^= 0xFF
        self.payloads[asset.url] = bytes(payload)
        record = self.record([asset])
        with self.assertLogs("fipd.collection.base", level="WARNING"):
            self.scraper.download_images(record, min_bytes=0)
        self.assertEqual(record.images, [])
        self.assertEqual(list(self.scraper.image_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
