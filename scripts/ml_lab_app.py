"""Standalone ML lab app, kept outside the minor project's page navigation."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st
from fipd.frontend.styles import CSS, header_html, page_header_html

from fipd.datasets.experiments import fingerprint
from fipd.models.classical import (
    DEFAULT_IMAGE_WEIGHTS, MobileNetFeatures, canonical_image, load_lab_artifact, predict_lab,
)

ROOT = Path(__file__).resolve().parents[1]
RUN_ROOT = ROOT / "data/processed/lab"
BUNDLED_MODELS = ROOT / "demo_bundle/models"

st.set_page_config(page_title="ML Lab · Promotion Lens", layout="wide")
st.markdown(f"<style>{CSS}</style>", unsafe_allow_html=True)
st.markdown(page_header_html("Model comparisons", "Machine learning lab",
            "Compare text, image and fused features. Explore model outputs alongside their training provenance."),
            unsafe_allow_html=True)
st.caption("TF-IDF text features · MobileNetV3 image features · Linear SVM and comparison models")


@st.cache_resource(show_spinner=False)
def cached_model(path, modified, internal):
    return load_lab_artifact(path, allow_internal=internal)


@st.cache_resource(show_spinner=False)
def cached_encoder(path, modified):
    return MobileNetFeatures(path)


@st.cache_resource(show_spinner=False)
def cached_ocr():
    from fipd.frontend.analysis import load_ocr

    return load_ocr(ROOT)


with st.sidebar:
    st.markdown(header_html(), unsafe_allow_html=True)
    st.header("Lab models")
    internal = st.checkbox("Show internal pilot models", value=False)
    purpose = "internal_smoke_only" if internal else "research"
    artifacts = sorted((RUN_ROOT / purpose).glob("*/models/*.joblib"), reverse=True)
    if internal and BUNDLED_MODELS.is_dir():
        artifacts = sorted(BUNDLED_MODELS.glob("*.joblib")) + artifacts
    if internal:
        st.warning("Provisional pilot data. Outputs and metrics are unvalidated.")
    st.caption("Train a new run from the terminal using scripts/10_train_lab.py.")

if not artifacts:
    st.info("No trained models are available for this mode. Research training requires reviewed "
            "fake and legitimate examples. The internal pilot can be enabled in the sidebar.")
    st.code(".\\.venv\\Scripts\\python.exe scripts/10_train_lab.py --arms text_only", language="powershell")
    st.stop()

def recommended_model_index(paths):
    for index, path in enumerate(paths):
        if path.parent == BUNDLED_MODELS and path.stem == "fused_linear_svm":
            return index
    for index, path in enumerate(paths):
        if path.stem == "fused_linear_svm":
            return index
    return 0


artifact_path = st.selectbox(
    "Trained model", artifacts,
    index=recommended_model_index(artifacts),
    format_func=lambda path: f"{path.parent.parent.name} / {path.stem.replace('_', ' ')}")
try:
    artifact = cached_model(str(artifact_path), artifact_path.stat().st_mtime_ns, internal)
except (ValueError, OSError, ImportError) as exc:
    st.error(str(exc))
    st.stop()

if internal:
    st.warning("This model was trained for internal evaluation. "
               "Its predictions and test metrics do not establish real-world detection accuracy.")

if internal:
    st.subheader("Three case demonstration")
    examples = {
        "Scam promotion": ("sample_fake_1.png", "fake"),
        "Genuine promotion": ("sample_real_with_text_1.webp", "real"),
        "Ambiguous promotion": ("sample_fake_3.png", "review"),
    }
    case = st.selectbox("Case", list(examples))
    sample_name, expected = examples[case]
    sample_path = ROOT / "test_samples" / sample_name
    st.image(str(sample_path), width=420)
    if st.button("Analyze this case"):
        from fipd.frontend.analysis import analyze_image

        with Image.open(sample_path) as source:
            sample_image = canonical_image(source)
        extracted = analyze_image(sample_image, root=ROOT, ocr_loader=cached_ocr)
        if extracted.get("error"):
            st.error(extracted["error"]["message"])
        else:
            text = extracted["raw_ocr"]
            st.write("OCR evidence", text or "No text extracted")
            run = (BUNDLED_MODELS if BUNDLED_MODELS.is_dir() else
                   RUN_ROOT / "internal_smoke_only/visual-source-holdout-20261113/models")
            predictions = []
            for name in ("text_only_linear_svm", "fused_linear_svm"):
                path = run / f"{name}.joblib"
                model = cached_model(str(path), path.stat().st_mtime_ns, True)
                encoder = None
                if model["arm"] == "fused":
                    weights = (ROOT / DEFAULT_IMAGE_WEIGHTS if (ROOT / DEFAULT_IMAGE_WEIGHTS).is_file()
                               else ROOT / "demo_bundle/weights" / DEFAULT_IMAGE_WEIGHTS.name)
                    encoder = cached_encoder(str(weights), weights.stat().st_mtime_ns)
                predictions.append(predict_lab(model, text, image=sample_image, encoder=encoder))
            first, second = st.columns(2)
            for column, name, result in zip((first, second), ("Text SVM", "Fused SVM"), predictions):
                column.metric(name, result["label"].title())
                column.caption(f"Decision margin: {result['score']:.3f}")
            verdict = "Needs review" if expected == "review" or predictions[0]["label"] != predictions[1]["label"] else predictions[1]["label"].title()
            st.write("Case state", verdict)
            st.caption("Demonstration samples are provisional. Verify claims against the original source.")

left, right = st.columns([1, 1])
with left:
    st.subheader("Try a promotion")
    upload = st.file_uploader("Promotion image", type=["png", "jpg", "jpeg", "webp"])
    image = None
    image_hash = ""
    if upload is not None:
        try:
            content = upload.getvalue()
            if len(content) > 10 * 1024**2:
                raise ValueError("Choose an image smaller than 10 MB.")
            with Image.open(io.BytesIO(content)) as source:
                if source.width * source.height > 12_000_000:
                    raise ValueError("Choose an image with at most 12 million pixels.")
                image = canonical_image(source)
            image_hash = hashlib.sha256(content).hexdigest()
            st.image(image, width="stretch")
        except (OSError, ValueError, UnidentifiedImageError) as exc:
            st.error(f"Image could not be read: {exc}")
    if st.session_state.get("lab_image_hash") != image_hash:
        st.session_state.lab_image_hash = image_hash
        st.session_state.lab_ocr_text = ""
        st.session_state.pop("lab_prediction", None)
    if st.button("Read image text", disabled=image is None):
        from fipd.frontend.analysis import analyze_image

        with st.spinner("Reading image text…"):
            extracted = analyze_image(image, root=ROOT, ocr_loader=cached_ocr)
        if extracted.get("error"):
            st.error(extracted["error"]["message"])
        else:
            st.session_state.lab_ocr_text = extracted["raw_ocr"]
            st.session_state.pop("lab_prediction", None)
    ocr_text = st.text_area("Text from the image", key="lab_ocr_text", height=140,
                            help="Use OCR above or paste an exact transcript of the promotion.")
    caption = st.text_area("Original caption (optional)", key="lab_caption", height=90)
    st.caption("Use the promotion's own wording. Fact-check verdicts and article titles are excluded.")
    input_id = fingerprint({"artifact": str(artifact_path), "modified": artifact_path.stat().st_mtime_ns,
                            "image": image_hash, "ocr": ocr_text, "caption": caption})
    if st.session_state.get("lab_prediction_id") != input_id:
        st.session_state.pop("lab_prediction", None)
    if st.button("Classify promotion", type="primary"):
        try:
            encoder = None
            if artifact["arm"] != "text_only":
                weights = (ROOT / DEFAULT_IMAGE_WEIGHTS if (ROOT / DEFAULT_IMAGE_WEIGHTS).is_file()
                           else ROOT / "demo_bundle/weights" / DEFAULT_IMAGE_WEIGHTS.name)
                encoder = cached_encoder(str(weights), weights.stat().st_mtime_ns)
            with st.spinner("Running the model…"):
                result = predict_lab(artifact, ocr_text, caption, image, encoder)
            st.session_state.lab_prediction = result
            st.session_state.lab_prediction_id = input_id
        except (ValueError, OSError, ImportError) as exc:
            st.error(str(exc))

with right:
    st.subheader("Model output")
    result = st.session_state.get("lab_prediction")
    if result:
        st.metric("Assigned class", "Fake" if result["label"] == "fake" else "Legitimate")
        if result["score_kind"] == "decision_margin":
            st.metric("SVM decision margin", f"{result['score']:.4f}")
            st.caption("A margin at or above zero selects fake. This value is not a probability.")
        else:
            st.metric("Fake class score", f"{result['score']:.1%}")
            st.caption("This class probability has not been calibrated on independent data.")
        st.caption("The assigned class is a model prediction; it does not establish authenticity.")
        st.download_button("Download prediction", json.dumps(result, indent=2, allow_nan=False),
                           file_name="lab-prediction.json", mime="application/json")
    else:
        st.info("Choose a trained model, supply its inputs, and classify the promotion.")
    st.write("**Features:**", artifact["arm"].replace("_", " "))
    st.write("**Classifier:**", artifact["model"].replace("_", " "))
    st.write("**Selected C / alpha:**", artifact["selected_value"])
    st.caption("Hyperparameters were selected using validation macro-F1. "
               "TF-IDF vocabulary and image scaling were fitted on training data only.")

with st.expander("Training run and model comparisons"):
    report_path = (ROOT / "demo_bundle/experiment.json" if artifact_path.parent == BUNDLED_MODELS
                   else artifact_path.parent.parent / "experiment.json")
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
        st.caption(report["notice"])
        st.write("Split sizes:", {name: sum(counts.values()) for name, counts in report["counts"].items()})
        columns = ["arm", "model", "validation_macro_f1", "macro_f1", "precision", "recall",
                   "test_n", "latency_p50_ms", "artifact_mb"]
        st.dataframe([{key: row[key] for key in columns} for row in report.get("results", [])],
                     hide_index=True, width="stretch")
        st.caption("macro_f1, precision and recall are measured on the held-out test split. "
                   "Latency includes feature extraction and classifier inference; it excludes "
                   "OCR, file decoding and model loading. Image models also need the shared "
                   "MobileNet weights (about 9.8 MB), in addition to each classifier artifact.")
        report_dir = ROOT / "demo_bundle" if artifact_path.parent == BUNDLED_MODELS else artifact_path.parent.parent
        detail_path = report_dir / "evaluations" / f"{artifact_path.stem}.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        st.write("**Confusion counts (fake is the positive class):**", detail["metrics"]["confusion"])
        st.download_button("Download comparison CSV",
                           (report_dir / "comparison.csv").read_bytes(),
                           file_name="lab-comparison.csv", mime="text/csv")
    except (OSError, ValueError, KeyError):
        st.info("A complete experiment report is not available for this artifact.")
