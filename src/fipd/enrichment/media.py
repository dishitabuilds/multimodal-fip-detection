"""Bounded local media decoding and explicitly prepared speech recognition."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import io
import math
from pathlib import Path
import time

import numpy as np

MEDIA_VERSION = "local-media-v1"
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".webm", ".mkv"}
DEFAULT_SPEECH_MODEL = Path("data/models/faster-whisper-base")
SPEECH_REPOSITORY = "Systran/faster-whisper-base"
SAMPLE_RATE = 16000


class MediaError(ValueError):
    """An actionable input, asset or decoding error."""


@dataclass(frozen=True)
class MediaLimits:
    max_bytes: int = 50 * 1024**2
    max_seconds: float = 120.0
    frame_interval: float = 5.0
    max_frames: int = 60
    max_pixels: int = 8_300_000
    max_decoded_frames: int = 20_000
    decode_seconds: float = 45.0

    def __post_init__(self):
        if any(not math.isfinite(value) or value <= 0 for value in vars(self).values()):
            raise MediaError("Media limits must be finite and positive.")
        if any(not isinstance(getattr(self, name), int) for name in (
                "max_bytes", "max_frames", "max_pixels", "max_decoded_frames")):
            raise MediaError("Media count and size limits must be integers.")
        if math.ceil(self.max_seconds / self.frame_interval) > self.max_frames:
            raise MediaError("Frame interval would exceed the maximum sampled frame count.")


@dataclass
class SampledFrame:
    timestamp: float
    image: object


@dataclass
class DecodedMedia:
    sha256: str
    duration: float
    has_audio: bool
    has_video: bool
    audio: np.ndarray | None = None
    frames: list[SampledFrame] = field(default_factory=list)


def validate_upload(data: bytes, filename: str, limits: MediaLimits):
    if not isinstance(data, bytes) or not data:
        raise MediaError("Choose a nonempty audio or video file.")
    if len(data) > limits.max_bytes:
        raise MediaError(f"Choose a file no larger than {limits.max_bytes / 1024**2:g} MB.")
    if Path(filename).suffix.lower() not in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS:
        raise MediaError("Unsupported file extension. Use MP3, WAV, M4A, OGG, FLAC, MP4, MOV, WebM or MKV.")


def decode_media(data: bytes, filename: str, *, limits: MediaLimits | None = None,
                 include_audio=True, include_video=True) -> DecodedMedia:
    """Decode bytes in memory. No external commands, files or network requests.

    Audio PTS are retained as silence/gaps in a 16 kHz mono timeline, so delayed
    speech and sampled video timestamps share the container playback origin.
    Unknown-duration streams are checked during decoding; overlong files fail
    rather than silently analyzing a prefix.
    """
    limits = limits or MediaLimits()
    validate_upload(data, filename, limits)
    try:
        import av
    except ImportError as exc:
        raise MediaError('Media decoding is unavailable. Install: pip install -e ".[media]"') from exc
    started = time.monotonic()

    def check_work(count):
        if count > limits.max_decoded_frames or time.monotonic() - started > limits.decode_seconds:
            raise MediaError("This file exceeds the decoding work limit. Use a shorter or smaller clip.")

    def check_time(seconds):
        if not math.isfinite(seconds) or seconds > limits.max_seconds + 0.1:
            raise MediaError(f"Choose a clip no longer than {limits.max_seconds:g} seconds.")

    try:
        # Explicit protocol allowlist also prevents nested network media sources.
        options = {"protocol_whitelist": "pipe", "threads": "2",
                   "format_whitelist": "mp3,wav,mov,ogg,flac,matroska,webm"}
        with av.open(io.BytesIO(data), options=options) as container:
            audio_streams = list(container.streams.audio)
            video_streams = [s for s in container.streams.video
                             if not (s.disposition & av.stream.Disposition.attached_pic)]
            if not audio_streams and not video_streams:
                raise MediaError("The file contains no supported audio or video stream.")
            origin = float(container.start_time or 0) / av.time_base
            duration = float(container.duration or 0) / av.time_base
            check_time(duration)
            audio_index = audio_streams[0].index if audio_streams else None
            video_index = video_streams[0].index if video_streams else None
            result = DecodedMedia(hashlib.sha256(data).hexdigest(), duration,
                                  bool(audio_streams), bool(video_streams))
        if include_audio and audio_index is not None:
            with av.open(io.BytesIO(data), options=options) as container:
                stream = container.streams[audio_index]
                stream.codec_context.thread_count = 2
                resampler = av.AudioResampler(format="fltp", layout="mono", rate=SAMPLE_RATE)
                samples = np.zeros(math.ceil(limits.max_seconds * SAMPLE_RATE), dtype=np.float32)
                cursor = 0

                def append(frame):
                    nonlocal cursor
                    values = frame.to_ndarray().reshape(-1)
                    position = round((float(frame.time) - origin) * SAMPLE_RATE) \
                        if frame.time is not None else cursor
                    end = position + len(values)
                    check_time(end / SAMPLE_RATE)
                    if end > 0:
                        left, right = max(0, position), min(len(samples), end)
                        if right > left:
                            samples[left:right] = values[left - position:right - position]
                        cursor = max(cursor, right)

                for count, frame in enumerate(container.decode(stream), 1):
                    check_work(count)
                    for converted in resampler.resample(frame):
                        append(converted)
                for converted in resampler.resample(None):
                    append(converted)
                result.audio = samples[:cursor]
                if not np.isfinite(result.audio).all():
                    raise MediaError("The decoded audio contains invalid samples.")
                result.duration = max(result.duration, cursor / SAMPLE_RATE)
        if include_video and video_index is not None:
            with av.open(io.BytesIO(data), options=options) as container:
                stream = container.streams[video_index]
                stream.codec_context.thread_count = 2
                next_sample = 0.0
                for count, frame in enumerate(container.decode(stream), 1):
                    check_work(count)
                    if frame.width * frame.height > limits.max_pixels:
                        raise MediaError("Video frames are too large. Resize the clip to 1080p or smaller.")
                    if frame.time is None:
                        raise MediaError("Video frames need playback timestamps for aligned evidence.")
                    timestamp = float(frame.time) - origin
                    check_time(timestamp)
                    result.duration = max(result.duration, timestamp)
                    if timestamp + 1e-6 < next_sample:
                        continue
                    if len(result.frames) >= limits.max_frames:
                        raise MediaError("Too many sampled frames; increase the frame interval.")
                    result.frames.append(SampledFrame(max(0.0, timestamp), frame.to_image()))
                    next_sample = timestamp + limits.frame_interval
        if include_audio and result.has_audio and (result.audio is None or not result.audio.size):
            raise MediaError("The audio stream could not be decoded into playable samples.")
        if include_video and result.has_video and not result.frames:
            raise MediaError("The video stream could not be decoded into readable frames.")
        return result
    except MediaError:
        raise
    except (av.error.FFmpegError, OSError, ValueError, OverflowError) as exc:
        raise MediaError("The media file is corrupt or uses an unsupported codec. Try WAV or MP4.") from exc


def load_speech_model(root: Path, model_dir: str | Path | None = None):
    """Inference is local-only, including the tokenizer and VAD assets."""
    path = Path(model_dir) if model_dir else Path(root) / DEFAULT_SPEECH_MODEL
    required = ("model.bin", "config.json", "tokenizer.json")
    if not all((path / name).is_file() for name in required):
        raise MediaError("Speech model is not prepared. Run scripts/14_analyze_media.py --prepare-model once.")
    try:
        from faster_whisper import WhisperModel

        model = WhisperModel(str(path.resolve()), device="cpu", compute_type="int8",
                             cpu_threads=4, num_workers=1, local_files_only=True)
        import json

        manifest = path / "preparation.json"
        model.fipd_provenance = json.loads(manifest.read_text(encoding="utf-8")) if manifest.is_file() \
            else {"local_directory": str(path.resolve()), "repository": "unrecorded"}
        return model
    except ImportError as exc:
        raise MediaError('Speech recognition is unavailable. Install: pip install -e ".[media]"') from exc
    except (RuntimeError, OSError, ValueError) as exc:
        raise MediaError("The prepared speech model could not be loaded. Check its local files.") from exc


def transcribe_audio(audio: np.ndarray, model, *, language: str | None = None) -> dict:
    if language not in {None, "en", "hi"}:
        raise MediaError("Choose automatic language detection, English or Hindi.")
    segments, info = model.transcribe(audio, language=language, beam_size=5, vad_filter=True,
                                      condition_on_previous_text=False)
    rows = []
    duration = len(audio) / SAMPLE_RATE
    for segment in segments:
        start, end, text = float(segment.start), float(segment.end), segment.text
        if (not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start
                or end > duration + 1 or not isinstance(text, str)):
            raise MediaError("Speech recognition returned invalid timestamps or text.")
        if text.strip():
            rows.append({"start": min(start, duration), "end": min(end, duration), "text": text.strip()})
        if len(rows) > 2000 or sum(len(row["text"]) for row in rows) > 100_000:
            raise MediaError("The transcription exceeds the text limit. Use a shorter clip.")
    return {"language": str(info.language), "segments": rows,
            "model": getattr(model, "fipd_provenance", {"repository": "caller_supplied"}),
            "compute_type": "int8", "device": "cpu"}
