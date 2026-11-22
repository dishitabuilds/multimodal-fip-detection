"""Promotion Lens: a local image/text research workbench.

Preview and OCR are independent of classifier checkpoints. Model inference is
explicitly selected, and the internal pilot is always labelled as unvalidated.
"""
from __future__ import annotations

from html import escape
import importlib
import io
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from fipd.frontend import analysis as analysis_service

if "multimodal" not in getattr(analysis_service, "MODES", ()):
    analysis_service = importlib.reload(analysis_service)
from fipd.frontend.styles import CSS, empty_state_html, header_html, page_header_html, render_navigation

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 12_000_000
MODE_LABELS = {
    "Lens Across": "multimodal",
    "Lens Invest": "svm",
    "Text Reader": "ocr",
}
PREVIOUS_MODE_LABELS = {
    "Cross-format detector (120-image model)": "Lens Across",
    "Classify promotion": "Lens Invest",
    "Extract text": "Text Reader",
    "Pilot demo": "Lens Invest",
    "Research model": "Lens Invest",
}
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
def cached_classifier(root, mode, checkpoint_stamp, config_stamp, asset_stamps=()):
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
        status = ("Review required" if result.get("review_reason") else
                  "Prediction complete" if result.get("label") else "Text extraction complete")
        st.markdown(f'<span class="status-pill ready">{status}</span>', unsafe_allow_html=True)

    scores = result.get("class_scores")
    if result.get("review_reason") in {"phishing_out_of_scope", "simulated_phishing_qr", "format_out_of_scope", "uncertain_prediction"} and not error:
        heading = ("Possible phishing · Review required"
                   if result.get("review_reason") in {"phishing_out_of_scope", "simulated_phishing_qr"}
                   else "Ambiguous evidence · Review required")
        st.markdown(f'<h3 class="result-heading">{heading}</h3>',
                    unsafe_allow_html=True)
        if result.get("review_reason") == "simulated_phishing_qr":
            st.warning("Educational phishing simulation detected. The QR code contains simulation text rather than a live malicious link.")
        elif result.get("review_reason") == "uncertain_prediction":
            cal_prob = result.get("calibrated_probability", 0.5)
            st.warning(f"The scam score is {cal_prob:.1%}, within the 45%–55% review range. Verify the sender and destination before deciding whether this is real or fake.")
        elif result.get("review_reason") == "format_out_of_scope":
            fmt_str = result.get("detected_format", "message").replace("_", " ")
            st.warning(f"Detected {fmt_str} format. The 47-image baseline model cannot reliably assess this format alone; review required.")
        else:
            st.warning("The classifier cannot establish whether this "
                       "message is real. Reward and action wording needs sender and link verification.")
        with st.expander("Model details"):
            if result.get("mode") == "multimodal":
                st.write("Model:", "120-image Multimodal Specialist")
                st.write("Format specialist:", result.get("detected_format", "general").replace("_", " ").title())
                st.write("Calibrated risk score:", f"{result.get('calibrated_probability', 0.5):.1%}")
                if result.get("decision_score") is not None:
                    st.write("Specialist margin:", f"{result['decision_score']:+.3f}")
                st.write("Preliminary label:", result.get("model_label", "unknown").title())
                st.caption("Decision withheld: sender and destination cannot be verified from image pixels alone.")
            else:
                st.write("Raw pilot prediction:", result.get("model_label", "unknown").title())
                st.write("Decision margin:", f"{result['decision_score']:+.3f}")
                if result.get("detected_format"):
                    st.write("Detected format:", result["detected_format"].replace("_", " ").title())
                if result.get("multimodal_assessment"):
                    mm = result["multimodal_assessment"]
                    st.write("Format specialist model:", mm.get("model_version", "v1"))
                    st.write("Calibrated risk score:", f"{mm.get('calibrated_probability', 0.5):.1%}")
                st.caption("This preliminary model was trained on investment promotions, not phishing messages.")
    elif result["mode"] == "multimodal" and not error and result.get("label"):
        label = "Fake / Scam" if result["label"] == "fake" else "Real / Benign"
        st.markdown(f'<h3 class="result-heading">Model prediction: {label}</h3>',
                    unsafe_allow_html=True)
        fmt_str = result.get("detected_format", "general").replace("_", " ").title()
        st.caption(f"120-image multimodal model · {fmt_str} specialist · Platt-calibrated probabilities.")
        if scores:
            st.markdown(
                '<div class="score-strip">'
                f'<div class="score-item"><div class="score-label">Scam probability</div>'
                f'<div class="score-value">{scores["fake"]:.1%}</div></div>'
                f'<div class="score-item"><div class="score-label">Benign probability</div>'
                f'<div class="score-value">{scores["real"]:.1%}</div></div>'
                f'<div class="score-item"><div class="score-label">Decision threshold</div>'
                f'<div class="score-value">{result.get("threshold", 0.5):.1%}</div></div></div>',
                unsafe_allow_html=True,
            )
        with st.expander("Model details"):
            st.write("Model:", "120-image Cross-Format Specialist")
            st.write("Format specialist:", fmt_str)
            st.write("Decision margin:", f"{result.get('decision_score', 0.0):+.3f}")
            st.write("Calibrated risk:", f"{result.get('calibrated_probability', 0.5):.1%}")
            st.write("Model version:", result.get("experiment_id", "multimodal_generalization_v1"))
            st.caption("Probabilities are calibrated using Platt scaling on held-out validation margins.")
    elif result["mode"] == "svm" and not error and result.get("label"):
        label = "Fake" if result["label"] == "fake" else "Real"
        st.markdown(f'<h3 class="result-heading">Model prediction: {label}</h3>',
                    unsafe_allow_html=True)
        st.caption("47-image pilot model · A content prediction, not a guarantee of investment safety.")
        with st.expander("Model details"):
            st.write("Model:", "Image + text Linear SVM")
            st.write("Experiment:", result["experiment_id"])
            st.write("Decision margin:", f"{result['decision_score']:+.3f}")
            if result.get("detected_format"):
                st.write("Detected format:", result["detected_format"].replace("_", " ").title())
            if result.get("multimodal_assessment"):
                mm = result["multimodal_assessment"]
                st.write("Format specialist risk:", f"{mm.get('calibrated_probability', 0.5):.1%}")
            st.caption("The signed margin selects Fake or Real. It is not a confidence percentage.")
    elif scores and not error:
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

    for warning in result.get("warnings", []):
        if result["mode"] not in {"svm", "multimodal"}:
            st.info(warning)

    if result.get("ocr") is not None:
        st.markdown("**Extracted text**")
        if result.get("raw_ocr"):
            st.code(result["raw_ocr"], language=None, wrap_lines=True)
        else:
            st.caption("No readable text was found. Try a clearer or more tightly cropped image.")
        if result.get("qr_payloads"):
            st.markdown("**QR code content**")
            for payload in result["qr_payloads"]:
                st.code(payload, language=None, wrap_lines=True)
            st.caption("Decoded locally; destinations are not opened or verified.")
        ocr = result["ocr"]
        line_count = len(ocr.get("lines", []))
        duration = result.get("durations_ms", {}).get("total", 0) / 1000
        st.caption(f"{line_count} text lines · {duration:.1f} s total analysis time")

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
            mode_label = next((k for k, v in MODE_LABELS.items() if v == result["mode"]), result["mode"])
            st.write("Mode:", mode_label)
            st.write("Model status:", result["model_status"].replace("_", " "))
            st.write("Input:", result.get("input"))
            if result.get("detected_format"):
                st.write("Detected layout format:", result["detected_format"].replace("_", " ").title())
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
    # Preserve the user's mode when its display name changes; clear stale results.
    if st.session_state.get("analysis_mode") not in MODE_LABELS:
        st.session_state.analysis_mode = PREVIOUS_MODE_LABELS.get(
            st.session_state.get("analysis_mode"), "Lens Across")
        clear_result()
    selected_mode = st.selectbox("Analysis mode", list(MODE_LABELS), key="analysis_mode")
    mode = MODE_LABELS[selected_mode]
    runtime = analysis_service.mode_runtime(ROOT, mode)
    if mode == "ocr":
        st.markdown('<span class="status-pill ready">Text extraction</span>', unsafe_allow_html=True)
        st.caption("Read English and Hindi text from the selected image. No classifier is used.")
    elif mode == "multimodal":
        status = "120-image Multimodal ready" if runtime["checkpoint_available"] else "Model files missing"
        st.markdown(f'<span class="status-pill experimental">{status}</span>', unsafe_allow_html=True)
        st.caption("Cross-format specialist for investment promotions, phishing email screenshots, "
                   "and SMS screenshots. Calibrated risk with human review triggers.")
    else:
        status = "47-image SVM ready" if runtime["checkpoint_available"] else "Model files missing"
        st.markdown(f'<span class="status-pill experimental">{status}</span>', unsafe_allow_html=True)
        st.caption("Predict Fake or Real using image text and visual features. "
                   "Preliminary pilot model; review the evidence alongside its prediction.")
    st.divider()
    with st.expander("How it works"):
        st.write("1. Choose one promotion image.")
        st.write("2. Add its original caption, if available.")
        st.write("3. Analyze the promotion to see its prediction and extracted evidence.")
        st.caption("Images are analyzed on this machine. Uploads are not added to the dataset. "
                   "OCR reads local model files; no automatic model downloads occur.")
    st.button("Clear workspace", key="reset_input", on_click=reset_input, use_container_width=True)
    st.caption("A Lightweight Multimodal Model for Detecting Fake Investment Promotions")
    st.caption("Minor project · Supervisor: Dr. Shivangi Surati")

st.markdown(page_header_html("Image + text review", "Look beyond the promise.",
            "Get a Fake or Real model prediction, then inspect the image, extracted wording "
            "and original caption together."), unsafe_allow_html=True)
render_navigation(ROOT, "image")

st.markdown('<div class="workflow-steps">'
            '<span class="workflow-step"><strong>01</strong> Choose a promotion</span>'
            '<span class="workflow-step"><strong>02</strong> Extract the evidence</span>'
            '<span class="workflow-step"><strong>03</strong> View prediction and evidence</span></div>', unsafe_allow_html=True)

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
    asset_stamps = tuple(stamp(path) for path in runtime.get("asset_paths", []))
    identity = ((source_id, analysis_service.input_fingerprint(image, caption, mode),
                 checkpoint_stamp, config_stamp, asset_stamps) if image is not None else None)
    if st.session_state.get("analysis_identity") != identity:
        clear_result()
    if mode != "ocr" and not runtime["checkpoint_available"]:
        st.warning("Classification model files are missing. Restore the required model files, "
                   "or choose Text Reader to inspect OCR evidence.")
    else:
        st.caption(f"Selected mode: {selected_mode}")
    run = st.button("Analyze promotion", key="analyze", type="primary", use_container_width=True,
                    disabled=image is None or (mode != "ocr" and not runtime["checkpoint_available"]))
    if run and image is not None:
        with st.spinner("Reading the image and assembling the evidence…"):
            result = analysis_service.analyze_image(
                image, caption, mode, ROOT, ocr_loader=lambda: cached_ocr(str(ROOT)),
                model_loader=lambda: cached_classifier(str(ROOT), mode, checkpoint_stamp, config_stamp,
                                                       asset_stamps),
            )
        st.session_state.analysis_result = result
        st.session_state.analysis_identity = identity
    if st.session_state.get("analysis_result"):
        st.markdown("[View analysis ↓](#analysis-result)")

with right, st.container(border=False, key="results_panel"):
    result = st.session_state.get("analysis_result")
    evidence_note = ("Evidence for the current image and caption." if result
                     else "Your prediction and extracted evidence will appear here.")
    st.markdown('<p class="section-eyebrow">02 / Result</p><h2 class="panel-title">Analysis result</h2>'
                f'<p class="panel-note">{evidence_note}</p>', unsafe_allow_html=True)
    if result:
        render_result(result)
    else:
        st.markdown(empty_state_html(), unsafe_allow_html=True)
        st.caption("Your analysis stays here until you change the image, caption or mode.")

st.markdown('<p class="report-footnote">Promotion Lens · Image and text research workspace '
            '· OCR extraction is separate from model classification.</p>', unsafe_allow_html=True)
