"""Real codec round trips plus media orchestration without network downloads."""
from __future__ import annotations

from fractions import Fraction
import io
import json
from pathlib import Path
from types import SimpleNamespace
import wave

import numpy as np
import pytest

from fipd.analysis import media as analysis
from fipd.enrichment import media


def wav_bytes(seconds=1, rate=8000):
    output = io.BytesIO()
    with wave.open(output, "wb") as handle:
        handle.setparams((1, 2, rate, 0, "NONE", "not compressed"))
        signal = (np.sin(np.arange(round(seconds * rate)) * 2 * np.pi * 220 / rate) * 8000).astype("<i2")
        handle.writeframes(signal.tobytes())
    return output.getvalue()


def video_bytes(seconds=3, *, audio=False, delay=0):
    av = pytest.importorskip("av")
    output = io.BytesIO()
    with av.open(output, "w", format="matroska") as container:
        video = container.add_stream("ffv1", rate=4)
        video.width, video.height, video.pix_fmt = 64, 48, "yuv420p"
        audio_stream = container.add_stream("pcm_s16le", rate=16000) if audio else None
        if audio_stream:
            audio_stream.layout = "mono"
        for index in range(seconds * 4):
            frame = av.VideoFrame.from_ndarray(np.full((48, 64, 3), index * 5, dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = index, Fraction(1, 4)
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        if audio_stream:
            for second in range(delay, seconds):
                frame = av.AudioFrame.from_ndarray(np.full((1, 16000), 4000, dtype=np.int16),
                                                   format="s16", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = 16000, second * 16000, Fraction(1, 16000)
                for packet in audio_stream.encode(frame):
                    container.mux(packet)
            for packet in audio_stream.encode():
                container.mux(packet)
    return output.getvalue()


def test_audio_is_resampled_with_actual_duration():
    pytest.importorskip("av")
    decoded = media.decode_media(wav_bytes(), "clip.wav")
    assert decoded.has_audio and not decoded.has_video
    assert decoded.audio.dtype == np.float32
    assert len(decoded.audio) == 16000
    assert decoded.duration == pytest.approx(1)
    assert np.abs(decoded.audio).max() > 0.1


def test_video_sampling_and_delayed_audio_share_the_playback_timeline():
    decoded = media.decode_media(video_bytes(audio=True, delay=1), "clip.mkv",
                                 limits=media.MediaLimits(frame_interval=2))
    assert [frame.timestamp for frame in decoded.frames] == [0, 2]
    assert decoded.frames[0].image.size == (64, 48)
    assert decoded.has_audio and decoded.has_video
    assert len(decoded.audio) == 48000
    assert np.abs(decoded.audio[:16000]).max() == 0
    assert np.abs(decoded.audio[16000:]).max() > 0.1


def test_corrupt_oversized_and_overlong_inputs_are_rejected():
    pytest.importorskip("av")
    with pytest.raises(media.MediaError, match="corrupt"):
        media.decode_media(b"invalid file", "clip.mp4")
    with pytest.raises(media.MediaError, match="no larger"):
        media.decode_media(wav_bytes(), "clip.wav", limits=media.MediaLimits(max_bytes=20))
    with pytest.raises(media.MediaError, match="no longer"):
        media.decode_media(wav_bytes(2), "clip.wav", limits=media.MediaLimits(max_seconds=1))
    with pytest.raises(media.MediaError, match="extension"):
        media.decode_media(wav_bytes(), "clip.txt")


def test_resource_limits_and_missing_weights(tmp_path):
    with pytest.raises(media.MediaError, match="positive"):
        media.MediaLimits(frame_interval=float("nan"))
    with pytest.raises(media.MediaError, match="frame count"):
        media.MediaLimits(frame_interval=0.1)
    with pytest.raises(media.MediaError, match="not prepared"):
        media.load_speech_model(tmp_path)
    with pytest.raises(media.MediaError, match="too large"):
        media.decode_media(video_bytes(), "clip.mkv", limits=media.MediaLimits(max_pixels=1))


def fake_speech(text="Buy now for guaranteed double returns."):
    class Model:
        def transcribe(self, audio, **kwargs):
            assert kwargs["vad_filter"] is True
            return iter([SimpleNamespace(start=0, end=0.9, text=text)]), SimpleNamespace(language="en")
    return Model()


def test_audio_wording_evidence_has_timestamps_and_no_fraud_verdict():
    result = analysis.analyze_media(wav_bytes(), "clip.wav", speech_loader=fake_speech)
    assert result["status"] == "complete"
    assert result["speech_status"] == "complete"
    assert result["frame_status"] == "no_video"
    finding = next(item for item in result["transcription"]["findings"] if item["label"] == "return_promise")
    assert (finding["time_start"], finding["time_end"]) == (0, 0.9)
    assert result["model"] is None and "label" not in result
    json.dumps(result, allow_nan=False)


def test_asr_boundaries_do_not_turn_an_investor_warning_into_a_promise():
    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(start=0, end=0.4, text="Avoid promoters who promise"),
                         SimpleNamespace(start=0.4, end=0.9, text="guaranteed returns on your investment.")]), SimpleNamespace(language="en")
    result = analysis.analyze_media(wav_bytes(), "clip.wav", speech_loader=Model)
    assert not any(item["label"] == "return_promise" for item in result["transcription"]["findings"])


def test_failed_speech_does_not_hide_video_evidence():
    class OCR:
        def read(self, path):
            assert Path(path).is_file()
            return SimpleNamespace(text="Buy now for guaranteed double returns.", error="")

    def unavailable():
        raise media.MediaError("Speech model is not prepared.")

    result = analysis.analyze_media(video_bytes(audio=True), "clip.mkv",
                                    speech_loader=unavailable, ocr_loader=OCR)
    assert result["status"] == "partial"
    assert result["speech_status"] == "failed"
    assert result["frame_status"] == "complete"
    assert result["frames"][0]["findings"]
    assert result["errors"][0]["channel"] == "speech"


def test_empty_audio_and_optional_channels():
    result = analysis.analyze_media(wav_bytes(), "clip.wav", speech_loader=lambda: fake_speech(""))
    assert result["speech_status"] == "no_speech"
    assert result["transcription"]["segments"] == []
    with pytest.raises(media.MediaError, match="Enable"):
        analysis.analyze_media(wav_bytes(), "clip.wav", speech=False, frames=False)


def test_chunks_cover_the_whole_input_without_silent_truncation():
    text = "Investment promotion words. " * 1000
    chunks = list(analysis.text_chunks(text, 600))
    covered = set()
    for chunk in chunks:
        assert chunk["text"] == text[chunk["start"]:chunk["end"]]
        assert len(chunk["text"]) <= 600
        covered.update(range(chunk["start"], chunk["end"]))
    assert covered == set(range(len(text)))


def test_existing_text_classifier_is_reused_for_every_window(monkeypatch):
    from fipd.models import classical

    seen = []

    def predict(artifact, ocr, caption, image, encoder):
        seen.append(ocr)
        assert len(ocr) + len(caption) + 7 <= 300
        return {"label": "fake", "score": 1.2, "score_kind": "decision_margin"}

    monkeypatch.setattr(classical, "predict_lab", predict)
    artifact = {"arm": "text_only", "model": "linear_svm", "purpose": "internal_smoke_only",
                "experiment_id": "fixture", "text_config": {"max_chars": 300,
                "fields": ["ocr_text", "claim_text"]}}
    result = analysis.analyze_media(wav_bytes(), "clip.wav", artifact=artifact,
                                    speech_loader=lambda: fake_speech("Buy investment now. " * 50))
    assert len(seen) > 1
    assert len(result["transcription"]["segments"][0]["predictions"]) == len(seen)
    assert result["model"]["purpose"] == "internal_smoke_only"


def test_invalid_asr_output_is_reported_as_failed():
    class Model:
        def transcribe(self, audio, **kwargs):
            return iter([SimpleNamespace(start=float("nan"), end=1, text="hello")]), SimpleNamespace(language="en")
    result = analysis.analyze_media(wav_bytes(), "clip.wav", speech_loader=Model)
    assert result["status"] == "no_evidence"
    assert result["speech_status"] == "failed"
    assert result["errors"]


def test_failed_classifier_preserves_the_transcript(monkeypatch):
    from fipd.models import classical

    def broken(*args):
        raise ValueError("Classifier could not score this segment.")

    monkeypatch.setattr(classical, "predict_lab", broken)
    artifact = {"arm": "text_only", "model": "linear_svm", "purpose": "research",
                "experiment_id": "fixture", "text_config": {"max_chars": 4000,
                "fields": ["ocr_text", "claim_text"]}}
    result = analysis.analyze_media(wav_bytes(), "clip.wav", artifact=artifact,
                                    speech_loader=fake_speech)
    assert result["status"] == "partial"
    assert result["speech_status"] == "complete"
    assert result["transcription"]["segments"][0]["text"].startswith("Buy now")
    assert result["errors"][0]["channel"] == "speech_model"


def test_failed_frame_ocr_preserves_the_speech():
    class BrokenOCR:
        def read(self, path):
            return SimpleNamespace(text="", error="OCR failed")

    result = analysis.analyze_media(video_bytes(audio=True), "clip.mkv",
                                    ocr_loader=BrokenOCR, speech_loader=fake_speech)
    assert result["speech_status"] == "complete"
    assert result["frame_status"] == "partial"
    assert result["frames"][0]["error"]
    assert result["transcription"]["segments"]
