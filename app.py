"""Promotion Lens: a local image/text research workbench.

Preview and OCR are independent of classifier checkpoints. Model inference is
explicitly selected, and the internal pilot is always labelled as unvalidated.
"""
from __future__ import annotations

from html import escape
import io
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from fipd.frontend import analysis as analysis_service
from fipd.frontend.styles import CSS, empty_state_html, header_html

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MODE_LABELS = {"Extract text": "ocr", "Pilot demo": "pilot", "Research model": "research"}
SAMPLES = {
    "sample_public_figure": {
        "title": "Social feed promotion", "file": "sample_fake_1.png",
        "context": "Bundled social-feed screenshot; source and outcome are not verified by this demo.",
    },
    "sample_news_page": {
        "title": "News-style promotion", "file": "sample_fake_2.png",
        "context": "Bundled article-style screenshot; publisher styling is not verification.",
    },
    "sample_fund_comparison": {
        "title": "Scheme comparison graphic", "file": "sample_real_with_text_1.webp",
        "context": "AMFI-branded educational material; demonstration asset, not a held-out test result.",
    },
    "sample_goal_planning": {
        "title": "Goal planning graphic", "file": "sample_real_with_text_2.webp",
        "context": "AMFI-branded educational material; demonstration asset, not a held-out test result.",
    },
}

st.set_page_config(page_title="Promotion Lens", layout="wide", initial_sidebar_state="auto")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)


def clear_result():
    st.session_state.pop("analysis_result", None)
    st.session_state.pop("analysis_identity", None)


def open_samples():
    st.session_state.input_source = "Sample gallery"
    clear_result()


def select_sample(key):
    st.session_state.selected_sample = key
    st.session_state.caption_input = ""
    clear_result()


def reset_input():
    st.session_state.upload_revision = st.session_state.get("upload_revision", 0) + 1
    st.session_state.input_source = "Upload"
    st.session_state.selected_sample = None
    st.session_state.caption_input = ""
    clear_result()


def stamp(path):
    if not path:
        return None
    try:
        stat = Path(path).stat()
        return stat.st_mtime_ns, stat.st_size
    except OSError:
        return None


@st.cache_resource(show_spinner=False)
def cached_ocr(root):
    return analysis_service.load_ocr(Path(root))


@st.cache_resource(show_spinner=False)
def cached_classifier(root, mode, checkpoint_stamp, config_stamp):
    return analysis_service.load_classifier(Path(root), mode)


def decode_image(data):
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Choose an image smaller than 10 MB.")
    with Image.open(io.BytesIO(data)) as source:
        if source.width * source.height > MAX_IMAGE_PIXELS:
            raise ValueError("Choose an image with at most 12 million pixels.")
        if source.format not in {"PNG", "JPEG", "WEBP"}:
            raise ValueError("Choose a PNG, JPG or WebP image.")
        return analysis_service.canonical_image(source)


def render_result(result):
    error = result.get("error")
    if error:
        st.error(error["message"])
        if error.get("details"):
            st.caption(error["details"])
    else:
        st.markdown('<span class="status-pill ready">Analysis complete</span>', unsafe_allow_html=True)

    if result["mode"] == "pilot":
        st.warning("Unvalidated pilot output. This model uses a 14-image provisional dataset "
                   "for an internal pipeline demonstration. Its scores do not establish accuracy.")
    for warning in result.get("warnings", []):
        # The permanent pilot notice above carries the full limitation.
        if not (result["mode"] == "pilot" and "pilot" in warning.lower()):
            st.info(warning)

    scores = result.get("class_scores")
    if scores:
        st.markdown('<h3 class="result-heading">Model output</h3>', unsafe_allow_html=True)
        label = "Fake" if result["label"] == "fake" else "Real"
        st.write(f"Assigned class: **{label}**")
        st.caption("Class names describe the model's training labels. "
                   "They are not an assessment of investment safety or regulatory approval.")
        st.markdown(
            '<div class="score-strip">'
            f'<div class="score-item"><div class="score-label">Fake class score</div>'
            f'<div class="score-value">{scores["fake"]:.1%}</div></div>'
            f'<div class="score-item"><div class="score-label">Real class score</div>'
            f'<div class="score-value">{scores["real"]:.1%}</div></div>'
            f'<div class="score-item"><div class="score-label">Decision threshold</div>'
            f'<div class="score-value">{result["threshold"]:.2f}</div></div></div>',
            unsafe_allow_html=True,
        )
    elif not error:
        st.markdown('<h3 class="result-heading">Text, ready to inspect</h3>', unsafe_allow_html=True)
        st.caption("Text extraction mode does not assign a fake or real label.")

    if result.get("ocr") is not None:
        st.markdown("**Extracted text**")
        if result.get("raw_ocr"):
            st.code(result["raw_ocr"], language=None, wrap_lines=True)
        else:
            st.caption("No readable text was found. Try a clearer or more tightly cropped image.")
        ocr = result["ocr"]
        line_count = len(ocr.get("lines", []))
        duration = result.get("durations_ms", {}).get("total", 0) / 1000
        st.caption(f"{line_count} text lines · {duration:.1f} s elapsed · Local OCR")

        st.markdown("**Text cues**")
        indicators = result.get("indicators", [])
        if indicators:
            st.caption("Descriptive matches from the image text and your caption; "
                       "each needs context and independent verification.")
            items = []
            for cue in indicators:
                matches = ", ".join(cue["matches"])
                items.append(f'<div class="cue-item"><p class="cue-title">{escape(cue["label"])}</p>'
                             f'<p class="cue-description">{escape(matches)}</p></div>')
            st.markdown('<div class="cue-list">' + "".join(items) + '</div>', unsafe_allow_html=True)
        else:
            st.caption("No configured text cues matched. This does not establish authenticity.")

        with st.expander("Analysis details"):
            st.write("Mode:", next(label for label, value in MODE_LABELS.items() if value == result["mode"]))
            st.write("Model status:", result["model_status"].replace("_", " "))
            st.write("Input:", result.get("input"))
            st.caption("English and Hindi OCR · The supplied caption and this image's OCR are used together.")
            st.code(result["input_fingerprint"], language=None, wrap_lines=True)
        st.download_button(
            "Download analysis", data=json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
            file_name="promotion-lens-analysis.json", mime="application/json", key="download_analysis",
            use_container_width=True,
        )
    st.markdown('<p class="report-footnote">This workspace supports research and human review. '
                'Check the original source and the underlying claims before drawing a conclusion.</p>',
                unsafe_allow_html=True)


st.session_state.setdefault("upload_revision", 0)
st.session_state.setdefault("selected_sample", None)
st.session_state.setdefault("caption_input", "")

with st.sidebar:
    st.markdown(header_html(), unsafe_allow_html=True)
    st.markdown('<p class="section-eyebrow">Workspace settings</p>', unsafe_allow_html=True)
    selected_mode = st.selectbox("Analysis mode", list(MODE_LABELS), key="analysis_mode")
    mode = MODE_LABELS[selected_mode]
    runtime = analysis_service.mode_runtime(ROOT, mode)
    if mode == "ocr":
        st.markdown('<span class="status-pill ready">Text extraction</span>', unsafe_allow_html=True)
        st.caption("Read English and Hindi text from the selected image. No classifier is used.")
    elif mode == "pilot":
        st.markdown('<span class="status-pill experimental">Internal pilot · Unvalidated</span>', unsafe_allow_html=True)
        st.warning("Internal pilot: a 14-image provisional dataset. Use this mode only to explore the pipeline, "
                   "not to measure detection accuracy.")
    else:
        status = "Checkpoint present" if runtime["checkpoint_available"] else "Checkpoint unavailable"
        st.markdown(f'<span class="status-pill pending">{status}</span>', unsafe_allow_html=True)
        st.caption("Uses a compatible image + text research checkpoint. "
                   "Compatibility is checked before inference.")
    st.divider()
    with st.expander("How it works"):
        st.write("1. Choose one promotion image.")
        st.write("2. Add its original caption, if available.")
        st.write("3. Extract the text and inspect the evidence.")
        st.caption("Images are analyzed on this machine. Uploads are not added to the dataset. "
                   "OCR reads local model files; no automatic model downloads occur.")
    st.button("Clear workspace", key="reset_input", on_click=reset_input, use_container_width=True)
    st.caption("A Lightweight Multimodal Model for Detecting Fake Investment Promotions")
    st.caption("Minor project · Supervisor: Dr. Shivangi Surati")

st.markdown('<p class="hero-eyebrow">Promotion Lens / Image + text review</p>'
            '<h1 class="hero-title">Look beyond the promise.</h1>'
            '<p class="hero-description">Bring an investment promotion into focus. '
            'Read the text in its image, add the original caption, and inspect the evidence in one place.</p>'
            '<div class="workflow-steps">'
            '<span class="workflow-step"><strong>01</strong> Choose a promotion</span>'
            '<span class="workflow-step"><strong>02</strong> Extract the evidence</span>'
            '<span class="workflow-step"><strong>03</strong> Review in context</span></div>', unsafe_allow_html=True)

left, right = st.columns([1, 1.05], gap="large")
with left, st.container(border=False, key="input_panel"):
    st.markdown('<p class="section-eyebrow">01 / Input</p><h2 class="panel-title">The promotion</h2>'
                '<p class="panel-note">Use a screenshot, advertisement or promotional graphic.</p>',
                unsafe_allow_html=True)
    source = st.radio("Image source", ["Upload", "Sample gallery"], key="input_source", horizontal=True)
    image, source_id = None, None
    if source == "Upload":
        uploaded = st.file_uploader("Upload a promotion image", type=["png", "jpg", "jpeg", "webp"],
                                    key=f"upload_{st.session_state.upload_revision}",
                                    help="PNG, JPG or WebP · Up to 10 MB and 12 million pixels")
        if uploaded is not None:
            try:
                image = decode_image(uploaded.getvalue())
                source_id = ("upload", uploaded.name)
            except (ValueError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
                st.error("This image could not be opened. Use a readable PNG, JPG or WebP "
                         "under 10 MB and 12 million pixels.")
        else:
            st.caption("No image handy? Start with one of the bundled examples.")
            st.button("Explore sample gallery", key="open_samples", on_click=open_samples,
                      use_container_width=True)
    else:
        st.caption("Examples are for exploring the workflow. They are not a held-out benchmark.")
        for row in (("sample_public_figure", "sample_news_page"),
                    ("sample_fund_comparison", "sample_goal_planning")):
            for column, key in zip(st.columns(2), row):
                with column:
                    st.button(SAMPLES[key]["title"], key=key, on_click=select_sample, args=(key,),
                              type="primary" if st.session_state.selected_sample == key else "secondary",
                              use_container_width=True)
        selected_sample = SAMPLES.get(st.session_state.selected_sample)
        if selected_sample:
            try:
                image = decode_image((ROOT / "test_samples" / selected_sample["file"]).read_bytes())
                source_id = ("sample", st.session_state.selected_sample)
                st.markdown(f'<p class="source-note">{escape(selected_sample["context"])}</p>',
                            unsafe_allow_html=True)
            except (ValueError, OSError):
                st.error("This bundled image is unavailable. Choose another example or upload an image.")
    if image is not None:
        preview = image.copy()
        preview.thumbnail((720, 430))
        st.image(preview, caption="Selected promotion", use_container_width=False)
        st.caption(f"{image.width:,} × {image.height:,} pixels")
    caption = st.text_area("Original caption (optional)", key="caption_input", height=100, max_chars=3000,
                           placeholder="Paste the text that accompanied this promotion…",
                           help="Use the original caption. Sample descriptions are not supplied to the model.")
    checkpoint_stamp, config_stamp = stamp(runtime["checkpoint_path"]), stamp(runtime["config_path"])
    identity = ((source_id, analysis_service.input_fingerprint(image, caption, mode),
                 checkpoint_stamp, config_stamp) if image is not None else None)
    if st.session_state.get("analysis_identity") != identity:
        clear_result()
    if mode != "ocr" and not runtime["checkpoint_available"]:
        st.warning("The selected model checkpoint is unavailable. Choose Extract text to inspect OCR evidence.")
    elif mode == "pilot":
        st.warning("Pilot demo · Unvalidated scores from a 14-image provisional dataset.")
    else:
        st.caption(f"Selected mode: {selected_mode}")
    run = st.button("Analyze promotion", key="analyze", type="primary", use_container_width=True,
                    disabled=image is None or (mode != "ocr" and not runtime["checkpoint_available"]))
    if run and image is not None:
        with st.spinner("Reading the image and assembling the evidence…"):
            result = analysis_service.analyze_image(
                image, caption, mode, ROOT, ocr_loader=lambda: cached_ocr(str(ROOT)),
                model_loader=lambda: cached_classifier(str(ROOT), mode, checkpoint_stamp, config_stamp),
            )
        st.session_state.analysis_result = result
        st.session_state.analysis_identity = identity
    if st.session_state.get("analysis_result"):
        st.markdown("[View analysis ↓](#a-closer-look)")

with right, st.container(border=False, key="results_panel"):
    result = st.session_state.get("analysis_result")
    evidence_note = ("Evidence for the current image and caption." if result
                     else "Extracted text and review cues will appear here.")
    st.markdown('<p class="section-eyebrow">02 / Evidence</p><h2 class="panel-title">A closer look</h2>'
                f'<p class="panel-note">{evidence_note}</p>', unsafe_allow_html=True)
    if result:
        render_result(result)
    else:
        st.markdown(empty_state_html(), unsafe_allow_html=True)
        st.caption("Your analysis stays here until you change the image, caption or mode.")

st.markdown('<p class="report-footnote">Promotion Lens · Image and text research workspace '
            '· OCR extraction is separate from model classification.</p>', unsafe_allow_html=True)
