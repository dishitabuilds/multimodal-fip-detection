"""Timestamped media evidence; no unvalidated aggregate video verdict."""
from __future__ import annotations

import math
from pathlib import Path
import time

from ..enrichment.media import (
    MEDIA_VERSION, MediaError, MediaLimits, decode_media, load_speech_model, transcribe_audio,
)
from .promotion_audit import audit_promotion


def text_chunks(text: str, max_chars: int = 1800):
    """Preserve all text, splitting at whitespace with a short context overlap."""
    if max_chars < 160:
        raise MediaError("The model text limit is too small for media segments.")
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        yield {"start": start, "end": end, "text": text[start:end]}
        if end == len(text):
            break
        start = max(start + 1, end - min(120, max_chars // 4))


def inspect_text(text: str) -> dict:
    findings = []
    # Overlap retains local negation/qualifiers when a long transcript is split.
    for chunk in text_chunks(text, 8000):
        audit = audit_promotion(chunk["text"])
        for finding in audit["findings"]:
            finding = {**finding, "start": finding["start"] + chunk["start"],
                       "end": finding["end"] + chunk["start"]}
            if finding not in findings:
                findings.append(finding)
    return {"findings": findings, "method": "English wording rules; factual claims remain unchecked"}


def analyze_media(data: bytes, filename: str, *, root: Path = Path("."), caption: str = "",
                  language: str | None = None, speech=True, frames=True,
                  limits: MediaLimits | None = None, speech_loader=None, ocr_loader=None,
                  artifact=None, encoder=None, progress=None) -> dict:
    """Run independent speech/frame channels and retain recoverable failures.

    Scores belong to individual text windows or sampled frames, never the
    complete recording. Still-image classifiers have no validated media accuracy.
    """
    if not isinstance(caption, str) or len(caption) > 2000:
        raise MediaError("The original caption must contain at most 2,000 characters.")
    if not speech and not frames:
        raise MediaError("Enable speech transcription or sampled-frame analysis.")
    if language not in {None, "en", "hi"}:
        raise MediaError("Choose automatic language detection, English or Hindi.")
    limits = limits or MediaLimits()
    started = time.perf_counter()
    if progress:
        progress("Decoding media")
    decoded = decode_media(data, filename, limits=limits, include_audio=speech, include_video=frames)
    result = {"version": MEDIA_VERSION, "status": "complete", "sha256": decoded.sha256,
              "filename": Path(filename).name, "duration_seconds": decoded.duration,
              "has_audio": decoded.has_audio, "has_video": decoded.has_video,
              "caption": caption, "caption_audit": inspect_text(caption) if caption.strip() else None,
              "speech_status": "skipped" if not speech else "no_audio",
              "frame_status": "skipped" if not frames else "no_video",
              "transcription": None, "frames": [], "errors": [], "model": None,
              "frame_interval_seconds": limits.frame_interval,
              "limitations": [
                  "Speech and OCR can misread amounts, names and negation; review the extracted text.",
                  "Sampled frames can miss brief claims, warnings and scene changes.",
                  "English wording cues are unvalidated and do not verify factual claims.",
                  "Still-image/text model outputs have not been validated on audio or video.",
                  "No whole-file authenticity verdict, voice-clone or face-manipulation detection is produced.",
              ]}
    if artifact:
        result["model"] = {key: artifact[key] for key in ("arm", "model", "purpose", "experiment_id")}

    def failure(channel, exc):
        result["errors"].append({"channel": channel, "message": str(exc)})
        result["status"] = "partial"

    def predictions(text, image=None):
        if artifact is None or (image is None and artifact["arm"] != "text_only"):
            return []
        from ..models.classical import predict_lab

        maximum = int(artifact["text_config"]["max_chars"])
        separator = artifact["text_config"].get("separator", " [SEP] ")
        available = maximum - len(caption) - len(separator)
        if artifact["arm"] != "image_only" and available < 160:
            raise MediaError("The caption leaves too little space for media text in this model.")
        if artifact["arm"] == "image_only":
            chunks = [{"start": 0, "end": 0, "text": ""}]
        else:
            if "ocr_text" not in artifact["text_config"]["fields"]:
                raise MediaError("This text model was not trained to accept extracted media text.")
            chunks = list(text_chunks(text, available)) if text else [{"start": 0, "end": 0, "text": ""}]
        if artifact["arm"] != "image_only" and not text.strip() and not caption.strip():
            return []
        return [{"text_start": chunk["start"], "text_end": chunk["end"],
                 **predict_lab(artifact, chunk["text"], caption, image, encoder)} for chunk in chunks]

    if speech and decoded.has_audio:
        try:
            if progress:
                progress("Transcribing speech")
            model = speech_loader() if speech_loader else load_speech_model(root)
            transcription = transcribe_audio(decoded.audio, model, language=language)
            # Audit neighboring ASR segments together so warning/negation context
            # is not lost merely because the recognizer inserted a boundary.
            combined, spans = "", []
            for segment in transcription["segments"]:
                start = len(combined)
                combined += segment["text"] + " "
                spans.append((start, len(combined) - 1, segment))
                segment["predictions"] = []
                try:
                    segment["predictions"] = predictions(segment["text"])
                except (ValueError, RuntimeError, OSError, ImportError) as exc:
                    failure("speech_model", exc)
            transcription["findings"] = inspect_text(combined)["findings"] if combined.strip() else []
            for finding in transcription["findings"]:
                overlaps = [seg for left, right, seg in spans
                            if left < finding["end"] and right > finding["start"]]
                finding["time_start"] = min(seg["start"] for seg in overlaps)
                finding["time_end"] = max(seg["end"] for seg in overlaps)
            result["transcription"] = transcription
            result["speech_status"] = "complete" if spans else "no_speech"
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            failure("speech", exc)
            result["speech_status"] = "failed"
    if frames and decoded.has_video:
        from ..frontend.analysis import analyze_image, load_ocr

        result["frame_status"] = "complete"
        try:
            ocr = ocr_loader() if ocr_loader else load_ocr(root)
        except (ValueError, RuntimeError, OSError, ImportError) as exc:
            failure("frames", exc)
            result["frame_status"] = "failed"
            ocr = None
        if ocr is not None:
            for index, sampled in enumerate(decoded.frames):
                if progress:
                    progress(f"Reading frame {index + 1} of {len(decoded.frames)}")
                extracted = analyze_image(sampled.image, root=root, ocr_loader=lambda: ocr)
                row = {"timestamp": sampled.timestamp, "ocr_text": extracted["raw_ocr"],
                       "findings": [], "predictions": [], "error": extracted.get("error")}
                if row["error"]:
                    failure("frame", MediaError(f"At {sampled.timestamp:.1f}s: {row['error']['message']}"))
                    result["frame_status"] = "partial"
                else:
                    row["findings"] = inspect_text(row["ocr_text"])["findings"]
                    try:
                        row["predictions"] = predictions(row["ocr_text"], sampled.image)
                    except (ValueError, RuntimeError, OSError, ImportError) as exc:
                        failure("frame_model", exc)
                        row["error"] = {"message": str(exc)}
                result["frames"].append(row)
    if result["transcription"] is None and not result["frames"] and not caption.strip():
        result["status"] = "no_evidence"
    result["elapsed_seconds"] = time.perf_counter() - started
    if not math.isfinite(result["elapsed_seconds"]):
        raise MediaError("The media timer returned an invalid duration.")
    return result
