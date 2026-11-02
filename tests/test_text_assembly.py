"""Prevent OCR from archive annotations and unrelated images reaching a model."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fipd.datasets.examples import assemble_text, selected_ocr_text
from fipd.models.config import TextConfig
from fipd.schema.records import FactCheckRecord, ImageAsset


def record(*images):
    return FactCheckRecord(source="test", source_id="1", url="https://example.test/1",
                           images=list(images))


def test_ocr_matches_selected_image_and_excludes_annotations():
    rec = record(
        ImageAsset(url="creative", local_path="creative.jpg", role="creative",
                   width=800, height=600, ocr_text="Invest for long term goals"),
        ImageAsset(url="unrelated", local_path="thumbnail.jpg", role="creative",
                   width=100, height=100, ocr_text="Unrelated screenshot text"),
        ImageAsset(url="stamp", local_path="stamp.jpg", role="annotated",
                   width=2000, height=2000, ocr_text="FALSE DEBUNKED"),
    )
    text = assemble_text(rec, TextConfig(transliterate=False))
    assert text == "Invest for long term goals"


def test_frontend_in_memory_asset_keeps_ocr():
    rec = record(ImageAsset(url="", ocr_text="Uploaded image text"))
    assert selected_ocr_text(rec) == "Uploaded image text"


def test_no_fallback_to_annotation_when_creative_has_no_ocr():
    rec = record(ImageAsset(url="creative", local_path="creative.jpg", role="creative"),
                 ImageAsset(url="stamp", role="annotated", ocr_text="FALSE"))
    assert selected_ocr_text(rec) == ""
