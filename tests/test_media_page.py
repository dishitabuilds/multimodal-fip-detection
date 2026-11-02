"""Actual Streamlit page state, upload handling and result invalidation."""
import io
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from fipd.analysis import media
from tests.test_media import wav_bytes

ROOT = Path(__file__).resolve().parents[1]


def test_page_loads_without_a_classifier_and_requires_an_upload():
    page = AppTest.from_file(str(ROOT / "pages/3_Audio_Video.py")).run()
    assert not page.exception
    assert page.button(key="media_analyze").disabled
    assert page.selectbox(key="media_model").value is None


def test_upload_result_survives_reruns_and_is_cleared_on_changes(monkeypatch):
    upload = io.BytesIO(wav_bytes())
    upload.name = "promotion.wav"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)

    def analyze(data, filename, **kwargs):
        return {"duration_seconds": 1, "elapsed_seconds": 0.2, "status": "complete",
                "speech_status": "complete", "frame_status": "no_video", "errors": [],
                "caption_audit": None, "transcription": {"language": "en", "segments": [
                    {"start": 0, "end": 1, "text": "Guaranteed double returns.", "predictions": []}],
                    "findings": []}, "frames": [], "limitations": []}

    monkeypatch.setattr(media, "analyze_media", analyze)
    page = AppTest.from_file(str(ROOT / "pages/3_Audio_Video.py")).run()
    assert not page.exception and not page.button(key="media_analyze").disabled
    page.button(key="media_analyze").click().run()
    assert not page.exception
    assert page.session_state["media_result"]["speech_status"] == "complete"
    page.run()
    assert "media_result" in page.session_state
    page.selectbox(key="media_language").select("Hindi").run()
    assert "media_result" not in page.session_state
    page.button(key="media_analyze").click().run()
    page.text_area(key="media_caption").set_value("A new caption").run()
    assert "media_result" not in page.session_state
    page.button(key="media_analyze").click().run()
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: None)
    page.run()
    assert not page.exception
    assert "media_result" not in page.session_state
    assert page.text_area(key="media_caption").value == ""


def test_upload_error_is_visible_without_stale_results(monkeypatch):
    upload = io.BytesIO(b"not a recording")
    upload.name = "broken.mp4"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    page = AppTest.from_file(str(ROOT / "pages/3_Audio_Video.py")).run()
    page.button(key="media_analyze").click().run()
    assert not page.exception
    assert page.error
    assert "media_result" not in page.session_state
