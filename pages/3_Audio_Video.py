"""Local speech and sampled-frame evidence with optional existing lab models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import streamlit as st

from fipd.analysis.media import analyze_media
from fipd.frontend.styles import CSS, header_html, page_header_html, navigation_html
from fipd.enrichment.media import (
    AUDIO_EXTENSIONS, VIDEO_EXTENSIONS, DEFAULT_SPEECH_MODEL, MediaLimits, load_speech_model,
)

ROOT = Path(__file__).resolve().parents[1]

st.set_page_config(page_title="Audio & Video · Promotion Lens", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown(page_header_html("Audio + video review", "Hear the claim. See the context.",
            "Turn spoken promotions into timestamped evidence. Read the transcript, inspect on-screen "
            "wording and keep the passages that need a closer look."), unsafe_allow_html=True)
st.markdown(navigation_html("media", st.session_state.get("lens_page_routes")), unsafe_allow_html=True)

st.caption("Files are processed locally. Up to 50 MB and 2 minutes per clip. "
           "Speech and OCR are automatic transcripts; check names, numbers and context.")


@st.cache_resource(show_spinner=False)
def cached_speech(modified):
    return load_speech_model(ROOT)


@st.cache_resource(show_spinner=False)
def cached_ocr():
    from fipd.frontend.analysis import load_ocr

    return load_ocr(ROOT)


@st.cache_resource(show_spinner=False)
def cached_classifier(path, modified, internal):
    from fipd.models.classical import load_lab_artifact

    return load_lab_artifact(path, allow_internal=internal)


@st.cache_resource(show_spinner=False)
def cached_encoder(path, modified):
    from fipd.models.classical import MobileNetFeatures

    return MobileNetFeatures(path)


def stamp(path):
    return path.stat().st_mtime_ns if path.is_file() else None


def timestamp(seconds):
    return f"{int(seconds // 60):02d}:{seconds % 60:04.1f}"


def show_prediction(prediction):
    value = prediction["score"]
    score = f"margin {value:.3f}" if prediction["score_kind"] == "decision_margin" else f"class score {value:.3f}"
    st.caption(f"Model: {prediction['label']} · {score} · {prediction['arm'].replace('_', ' ')}. "
               "This is a segment/frame prediction with unvalidated media accuracy.")


with st.sidebar:
    st.markdown(header_html(), unsafe_allow_html=True)
    st.header("Media analysis")
    speech = st.checkbox("Transcribe speech", value=True, key="media_speech")
    frames = st.checkbox("Read video frames", value=True, key="media_frames")
    language = st.selectbox("Speech language", ["Automatic", "English", "Hindi"], key="media_language")
    interval = st.selectbox("Sample a frame every", [5, 2, 10],
                            format_func=lambda value: f"{value} seconds", key="media_interval")
    st.caption("Sampling can miss brief claims or warnings. Shorten the interval for rapidly changing clips.")
    speech_path = ROOT / DEFAULT_SPEECH_MODEL
    speech_stamp = stamp(speech_path / "model.bin")
    if speech and speech_stamp is None:
        st.info("Speech recognition needs a one-time model download from the terminal.")
        st.code('.\\.venv\\Scripts\\python.exe scripts/14_analyze_media.py --prepare-model', language="powershell")
    elif speech:
        st.markdown('<span class="status-pill ready">Speech model files available</span>', unsafe_allow_html=True)
    internal = st.checkbox("Show internal pilot classifiers", value=False, key="media_internal")
    purpose = "internal_smoke_only" if internal else "research"
    artifacts = sorted((ROOT / "data/processed/lab" / purpose).glob("*/models/*.joblib"), reverse=True)
    artifact_path = st.selectbox(
        "Optional classifier", [None, *artifacts], key="media_model",
        format_func=lambda value: "Extract and audit wording" if value is None
        else f"{value.parent.parent.name} / {value.stem.replace('_', ' ')}")
    if internal:
        st.warning("Pilot classifiers use 14 provisional images. Their predictions are unvalidated.")
    st.caption("Audio segments use text-only classifiers. Image and fused classifiers apply to sampled video frames.")

st.markdown('<div class="workflow-steps">'
            '<span class="workflow-step"><strong>01</strong> Upload a recording</span>'
            '<span class="workflow-step"><strong>02</strong> Extract speech + text</span>'
            '<span class="workflow-step"><strong>03</strong> Review timestamped evidence</span></div>',
            unsafe_allow_html=True)
input_column, evidence_column = st.columns([1, 1.1], gap="large")
input_panel = input_column.container(key="input_panel")
evidence_panel = evidence_column.container(key="results_panel")
with input_panel:
    st.markdown('<p class="section-eyebrow">01 / Input</p><h2 class="panel-title">The recording</h2>'
                '<p class="panel-note">A voice message, promotional recording or short video.</p>',
                unsafe_allow_html=True)
with input_panel:
    upload = st.file_uploader("Upload an audio or video promotion",
                              type=sorted(ext[1:] for ext in AUDIO_EXTENSIONS | VIDEO_EXTENSIONS),
                              key="media_upload")
content = upload.getvalue() if upload is not None else b""
digest = hashlib.sha256(content).hexdigest() if content else ""
filename = upload.name if upload is not None else ""
if st.session_state.get("media_source") != (digest, filename):
    st.session_state.media_source = (digest, filename)
    st.session_state.media_caption = ""
    st.session_state.pop("media_result", None)
valid_upload = bool(content) and len(content) <= 50 * 1024**2
if content:
    if not valid_upload:
        input_panel.error("Choose a file no larger than 50 MB.")
    elif Path(filename).suffix.lower() in VIDEO_EXTENSIONS:
        input_panel.video(content, format="video/webm" if filename.lower().endswith(".webm") else "video/mp4")
        input_panel.caption("Preview playback depends on your browser's codec support. Extraction uses the local decoder.")
    else:
        audio_type = {".mp3": "audio/mpeg", ".wav": "audio/wav", ".m4a": "audio/mp4",
                      ".ogg": "audio/ogg", ".flac": "audio/flac"}[Path(filename).suffix.lower()]
        input_panel.audio(content, format=audio_type)
caption = input_panel.text_area("Original caption (optional)", max_chars=2000, key="media_caption",
                               height=120, placeholder="Paste the text that accompanied this recording…")
is_audio = Path(filename).suffix.lower() in AUDIO_EXTENSIONS
if is_audio and not speech:
    input_panel.info("Enable Transcribe speech in the sidebar to analyze an audio file.")
input_panel.caption("MP3 · WAV · M4A · OGG · FLAC / MP4 · MOV · WEBM · MKV. Up to 50 MB / 2 minutes.")
identity = (digest, filename, caption, speech, frames, language, interval, internal,
            str(artifact_path), stamp(artifact_path) if artifact_path else None, speech_stamp)
if st.session_state.get("media_result_id") != identity:
    st.session_state.pop("media_result", None)
if input_panel.button("Analyze audio / video", type="primary", key="media_analyze", use_container_width=True,
                      disabled=not valid_upload or not (speech or (frames and not is_audio))):
    st.session_state.pop("media_result", None)
    status = input_panel.empty()
    try:
        artifact = encoder = None
        if artifact_path:
            artifact = cached_classifier(str(artifact_path), stamp(artifact_path), internal)
            if artifact["arm"] != "text_only":
                from fipd.models.classical import DEFAULT_IMAGE_WEIGHTS

                weights = ROOT / DEFAULT_IMAGE_WEIGHTS
                encoder = cached_encoder(str(weights), stamp(weights))
        with st.spinner("Extracting and analyzing media…"):
            result = analyze_media(
                content, filename, root=ROOT, caption=caption, speech=speech, frames=frames,
                language={"Automatic": None, "English": "en", "Hindi": "hi"}[language],
                limits=MediaLimits(frame_interval=float(interval)),
                speech_loader=lambda: cached_speech(speech_stamp), ocr_loader=cached_ocr,
                artifact=artifact, encoder=encoder, progress=status.info)
        st.session_state.media_result = result
        st.session_state.media_result_id = identity
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        input_panel.error(str(exc))
    finally:
        status.empty()

with evidence_panel:
    st.markdown('<p class="section-eyebrow">02 / Evidence</p><h2 class="panel-title">Inside the recording</h2>'
                '<p class="panel-note">Speech, wording cues and sampled on-screen text.</p>', unsafe_allow_html=True)
result = st.session_state.get("media_result")
with evidence_panel:
    if result:
        st.markdown('<span class="status-pill ready">Evidence extracted</span>' if result['status'] == 'complete'
                    else '<span class="status-pill experimental">Review extraction coverage</span>', unsafe_allow_html=True)
        duration_metric, speech_metric, frames_metric = st.columns(3)
        duration_metric.metric("Duration", f"{result['duration_seconds']:.1f} s")
        speech_metric.metric("Segments", len((result.get('transcription') or {}).get('segments', [])))
        frames_metric.metric("Frames", len(result.get('frames', [])))
        st.caption(f"Duration: {result['duration_seconds']:.1f}s · Processing: {result['elapsed_seconds']:.1f}s · "
                   f"Speech: {result['speech_status']} · Frames: {result['frame_status']}")
        for error in result["errors"]:
            st.warning(f"{error['channel']}: {error['message']}")
        if result["status"] == "no_evidence":
            st.info("No text evidence was extracted. Check the selected channels and any errors above.")
        if result["caption_audit"]:
            with st.expander("Original caption wording"):
                for finding in result["caption_audit"]["findings"]:
                    st.write(f"{finding['label'].replace('_', ' ')}: {finding['quote']}")
        transcript = result["transcription"]
        if transcript:
            st.subheader("Speech transcript")
            st.caption(f"Recognized language: {transcript['language']}. Wording flags below use English rules.")
            plain_transcript = "\n".join(
                f"[{timestamp(segment['start'])}–{timestamp(segment['end'])}] {segment['text']}"
                for segment in transcript['segments'])
            st.download_button("Download transcript", plain_transcript, file_name="promotion-transcript.txt",
                               mime="text/plain", key="media_transcript_download")
            for segment in transcript["segments"]:
                st.write(f"**{timestamp(segment['start'])}–{timestamp(segment['end'])}**")
                st.write(segment["text"])
                for prediction in segment["predictions"]:
                    show_prediction(prediction)
            for finding in transcript["findings"]:
                st.info(f"{timestamp(finding['time_start'])}–{timestamp(finding['time_end'])} · "
                        f"{finding['label'].replace('_', ' ')}: {finding['quote']}")
        if result["frames"]:
            st.subheader("Sampled video frames")
            for frame in result["frames"]:
                with st.expander(f"{timestamp(frame['timestamp'])} · {len(frame['findings'])} wording cue(s)"):
                    st.write(frame["ocr_text"] or "No text recognized in this sampled frame.")
                    if frame["error"]:
                        st.warning(frame["error"]["message"])
                    for finding in frame["findings"]:
                        st.info(f"{finding['label'].replace('_', ' ')}: {finding['quote']}")
                    for prediction in frame["predictions"]:
                        show_prediction(prediction)
        st.caption("No whole-file authenticity verdict is produced. Factual claims, cloned voices and manipulated faces remain unassessed.")
        st.download_button("Download media evidence JSON", json.dumps(result, ensure_ascii=False, indent=2,
                           allow_nan=False), file_name="media-evidence.json", mime="application/json")
        with st.expander("Coverage and limitations"):
            for limitation in result["limitations"]:
                st.write(limitation)
    else:
        st.markdown('<div class="evidence-empty media-empty"><div class="evidence-icon" aria-hidden="true">'
                    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">'
                    '<path d="M4 10v4m4-7v10m4-14v18m4-14v10m4-7v4"/></svg></div>'
                    '<div class="result-heading">Every claim has a context.</div>'
                    '<p class="result-description">Upload a clip to see timestamped speech, '
                    'on-screen text and wording cues together.</p></div>', unsafe_allow_html=True)
        st.caption("Wording analysis works without a trained classifier.")
