"""Actual Streamlit page state, upload handling and result invalidation."""
import io
from pathlib import Path

import pytest

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest
from fipd.analysis import media
from tests.test_media import wav_bytes

ROOT = Path(__file__).resolve().parents[1]


def upload_page():
    page = AppTest.from_file(str(ROOT / "pages/3_Audio_Video.py"), default_timeout=30)
    page.session_state["media_input_source"] = "Upload"
    return page.run()


def test_page_defaults_to_playable_demo_without_a_classifier():
    page = AppTest.from_file(str(ROOT / "pages/3_Audio_Video.py"), default_timeout=30).run()
    assert not page.exception
    assert page.radio(key="media_input_source").value == "Demo video"
    assert not page.button(key="media_analyze").disabled
    assert page.get("video")
    assert page.selectbox(key="media_model").value is None
    assert "media_result" not in page.session_state
    assert any("Fictional classroom" in item.value for item in page.info)
    page.radio(key="media_input_source").set_value("Demo audio").run()
    assert page.get("audio") and not page.button(key="media_analyze").disabled
    page.checkbox(key="media_speech").uncheck().run()
    assert page.button(key="media_analyze").disabled
    page.radio(key="media_input_source").set_value("Upload").run()
    assert page.button(key="media_analyze").disabled


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
    page = upload_page()
    assert not page.exception and not page.button(key="media_analyze").disabled
    page.button(key="media_analyze").click().run()
    assert not page.exception
    assert page.session_state["media_result"]["speech_status"] == "complete"
    transcript_download = next(item for item in page.get("download_button")
                               if item.proto.label == "Download transcript")
    assert transcript_download.proto.url
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
    page = upload_page()
    page.button(key="media_analyze").click().run()
    assert not page.exception
    assert page.error
    assert "media_result" not in page.session_state


def test_audio_upload_needs_speech_even_when_video_frames_are_enabled(monkeypatch):
    upload = io.BytesIO(wav_bytes())
    upload.name = "promotion.wav"
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: upload)
    page = upload_page()
    assert not page.button(key="media_analyze").disabled
    page.checkbox(key="media_speech").uncheck().run()
    assert not page.exception
    assert page.checkbox(key="media_frames").value
    assert page.button(key="media_analyze").disabled
    assert any("Enable Transcribe speech" in item.value for item in page.info)


def test_demo_runs_normal_analysis_and_source_switch_clears_result(monkeypatch):
    calls = []

    def analyze(data, filename, **kwargs):
        calls.append((data, filename))
        return {"duration_seconds": 18, "elapsed_seconds": 0.2, "status": "complete",
                "speech_status": "complete", "frame_status": "complete", "errors": [],
                "caption_audit": None, "transcription": {"language": "en", "segments": [
                    {"start": 5, "end": 9, "text": "Guaranteed monthly returns.", "predictions": []}],
                    "findings": []}, "frames": [], "limitations": []}

    monkeypatch.setattr(media, "analyze_media", analyze)
    page = AppTest.from_file(str(ROOT / "pages/3_Audio_Video.py"), default_timeout=30).run()
    assert not calls
    page.button(key="media_analyze").click().run()
    assert not page.exception
    assert calls == [((ROOT / "test_samples/media/promotion_demo.mp4").read_bytes(), "promotion_demo.mp4")]
    assert "media_result" in page.session_state
    assert any(item.proto.label == "Download transcript" for item in page.get("download_button"))
    page.run()
    assert "media_result" in page.session_state and len(calls) == 1
    page.radio(key="media_input_source").set_value("Upload").run()
    assert "media_result" not in page.session_state
    assert page.button(key="media_analyze").disabled


def test_missing_demo_keeps_upload_available(monkeypatch):
    original = Path.read_bytes

    def read(path):
        if path.name == "promotion_demo.mp4":
            raise FileNotFoundError(path)
        return original(path)

    monkeypatch.setattr(Path, "read_bytes", read)
    page = AppTest.from_file(str(ROOT / "pages/3_Audio_Video.py"), default_timeout=30).run()
    assert not page.exception and page.error
    assert page.button(key="media_analyze").disabled
    page.radio(key="media_input_source").set_value("Upload").run()
    assert not page.exception and not page.error
