"""Inspect misleading framing separately from the binary authenticity model."""
from __future__ import annotations

from datetime import date
import hashlib
import io
import json
from pathlib import Path

from PIL import Image, UnidentifiedImageError
import streamlit as st

from fipd.analysis.promotion_audit import AUDIT_VERSION, audit_promotion

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = {
    "Write your own": "",
    "Hypothetical promotion with a return promise": (
        "Revenue grew 40% last year. Buy this stock now for guaranteed double returns."
    ),
    "Hypothetical balanced opinion": (
        "Revenue grew 40% last year. The share price could rise if margins improve, "
        "but this is uncertain. You could lose money. Past performance does not "
        "guarantee future returns."
    ),
    "Hypothetical investor warning": (
        "Avoid promoters who promise guaranteed returns. Do not trust claims "
        "that a stock will double. Investments are subject to market risk."
    ),
}

st.set_page_config(page_title="Promotion audit · Promotion Lens", layout="wide")
st.title("Promotion framing and evidence audit")
st.write("Correct facts can still appear in a selective or overly promising promotion. "
         "Inspect its wording and record claim evidence separately.")
st.caption("English wording rules with reviewer-supplied evidence. "
           "This page works independently of the binary fake or legitimate model.")


@st.cache_resource(show_spinner=False)
def cached_ocr():
    from fipd.frontend.analysis import load_ocr

    return load_ocr(ROOT)


@st.cache_resource(show_spinner=False)
def cached_framing_model(path, modified):
    from fipd.training.framing import load_framing_artifact

    return load_framing_artifact(path)


def fingerprint(payload):
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False)
                          .encode("utf-8")).hexdigest()


with st.sidebar:
    st.header("What this audit covers")
    st.write("Wording cues identify passages to review. Evidence entries retain the "
             "reviewer's source, date and reasoning for individual claims.")
    st.caption("No share-price forecast, suitability rating or finding of fraud is produced. "
               "Influencer popularity and missing disclosures do not establish payment.")
    st.caption("For Hindi or other languages, translate and review the transcript manually; "
               "these English rules have no validated multilingual accuracy.")

    st.subheader("Learned framing model")
    model_paths = sorted((ROOT / "data/processed/framing/research").glob(
        "*/models/framing_linear_svm.joblib"), reverse=True)
    learned_path = None
    if model_paths:
        if st.checkbox("Include learned framing flags", value=False):
            learned_path = st.selectbox("Reviewed-data model", model_paths,
                                        format_func=lambda path: path.parent.parent.name)
            st.caption("Uses local models trained from attributed review annotations. "
                       "Learned labels remain separate from the evidence review.")
    else:
        st.caption("No reviewed-data framing model has been trained yet. "
                   "The wording and evidence audit is available below.")

st.subheader("Promotion inputs")
example = st.selectbox("Try a hypothetical example", list(EXAMPLES), key="audit_example")
if st.button("Load example", disabled=example == "Write your own", key="audit_load_example"):
    st.session_state.audit_text = EXAMPLES[example]
    st.session_state.audit_caption = ""
    st.session_state.audit_evidence = "[]"
    st.session_state.pop("promotion_audit", None)

upload = st.file_uploader("Promotion image (optional)", type=["png", "jpg", "jpeg", "webp"],
                          key="audit_image")
image = None
image_hash = ""
if upload is not None:
    try:
        content = upload.getvalue()
        image_hash = hashlib.sha256(content).hexdigest()
        if len(content) > 10 * 1024**2:
            raise ValueError("Choose an image smaller than 10 MB.")
        with Image.open(io.BytesIO(content)) as source:
            if source.width * source.height > 12_000_000:
                raise ValueError("Choose an image with at most 12 million pixels.")
            from fipd.frontend.analysis import canonical_image

            image = canonical_image(source)
        st.image(image, width="stretch")
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        st.error(f"Image could not be read: {exc}")

old_image_hash = st.session_state.get("audit_image_hash", "")
if old_image_hash != image_hash:
    st.session_state.audit_image_hash = image_hash
    st.session_state.audit_text = ""
    st.session_state.audit_caption = ""
    st.session_state.audit_evidence = "[]"
    st.session_state.pop("promotion_audit", None)
if st.button("Read image text", disabled=image is None, key="audit_read_image"):
    from fipd.frontend.analysis import analyze_image

    with st.spinner("Reading image text…"):
        extracted = analyze_image(image, root=ROOT, ocr_loader=cached_ocr)
    if extracted.get("error"):
        st.error(extracted["error"]["message"])
    else:
        st.session_state.audit_text = extracted["raw_ocr"]
        st.session_state.pop("promotion_audit", None)

text = st.text_area("Promotion transcript", key="audit_text", height=170,
                    help="Paste the exact promotion wording or read an image above. "
                         "Check OCR before auditing. Do not include a fact-check verdict.")
caption = st.text_area("Original caption (optional)", key="audit_caption", height=90)
as_of = st.date_input("Evidence cutoff date", value=date.today(), key="audit_as_of",
                      help="Use the promotion's publication date to review what evidence "
                           "was available at that time. Later evidence is rejected.")

with st.expander("Add reviewer claim evidence (optional)"):
    st.write("Enter a JSON list. Each entry must quote an exact span from the transcript "
             "or caption and name the human reviewer. Sources are recorded as supplied; "
             "the app does not open or independently verify them.")
    st.caption("Status may be supported, contradicted or insufficient_evidence. "
               "The source publication date must be on or before the cutoff. "
               "A supported historical fact does not support a future return promise.")
    st.code(json.dumps([{
        "claim_text": "Revenue grew 40% last year.",
        "field": "text",
        "status": "supported",
        "source_url": "https://example.com/annual-report",
        "source_title": "Replace with the actual filing title",
        "published_at": "2026-04-01",
        "reviewed_by": "Replace with the reviewer's name",
        "rationale": "Compare the exact company, period, units and reported figure.",
        "context": "Record relevant sourced context and explain why it matters."
    }], indent=2), language="json")
    evidence_text = st.text_area("Reviewer evidence JSON", value="[]", height=180,
                                key="audit_evidence")

as_of_text = as_of.isoformat() if isinstance(as_of, date) else None
input_id = fingerprint({"text": text, "caption": caption, "image": image_hash,
                        "as_of": as_of_text, "evidence": evidence_text,
                        "version": AUDIT_VERSION,
                        "learned_model": str(learned_path) if learned_path else None,
                        "learned_modified": learned_path.stat().st_mtime_ns if learned_path else None})
if st.session_state.get("promotion_audit_input_id") != input_id:
    st.session_state.pop("promotion_audit", None)

if st.button("Audit promotion", type="primary", key="audit_run"):
    try:
        if as_of_text is None:
            raise ValueError("Choose an evidence cutoff date before running the audit.")
        evidence = json.loads(evidence_text)
        if not isinstance(evidence, list):
            raise ValueError("Reviewer evidence JSON must be a list of claim entries.")
        result = audit_promotion(text, caption, as_of=as_of_text, evidence=evidence)
        if learned_path:
            try:
                from fipd.training.framing import predict_framing

                learned_model = cached_framing_model(str(learned_path),
                                                     learned_path.stat().st_mtime_ns)
                result["learned_framing"] = predict_framing(learned_model, text, caption)
            except (ValueError, OSError, ImportError) as exc:
                result["learned_model_error"] = str(exc)
        st.session_state.promotion_audit = result
        st.session_state.promotion_audit_input_id = input_id
    except (ValueError, TypeError) as exc:
        st.error(f"Audit could not run: {exc}")

result = st.session_state.get("promotion_audit")
if result:
    st.subheader("Framing observations")
    findings = result.get("findings", [])
    if not findings:
        st.info("No wording cues were found by these rules. This does not establish "
                "that the promotion is authentic, balanced or suitable.")
    for finding in findings:
        st.write(f"**{finding['label'].replace('_', ' ').capitalize()}**")
        st.text(finding["quote"])
        st.write(finding["reason"])
        st.caption(f"Exact source: {finding['field']} · characters "
                   f"{finding['start']}–{finding['end']} (zero based, end excluded)")
    for observation in result.get("observations", []):
        st.write(observation)

    if "learned_framing" in result:
        learned = result["learned_framing"]
        st.subheader("Learned framing patterns")
        st.write("Selected labels:", learned["labels"] or "None")
        st.write("SVM decision margins:", learned["margins"])
        st.caption(learned["notice"])
        st.caption("Margins are not probabilities. These learned patterns do not "
                   "establish whether a statement is true or explain a missing fact.")
    if result.get("learned_model_error"):
        st.warning("The learned model could not run: " + result["learned_model_error"])

    st.subheader("Reviewer claim evidence")
    claims = result.get("claims", [])
    if not claims:
        st.info("No reviewer evidence was supplied. Facts have not been checked.")
    for claim in claims:
        st.text(claim["claim_text"])
        st.write(f"**Reviewer status:** {claim['status'].replace('_', ' ')}")
        st.write(claim["rationale"])
        if claim.get("context"):
            st.write("**Reviewer context:**", claim["context"])
        st.write("**Supplied source:**", claim["source_title"])
        st.write(claim["source_url"])
        st.caption(f"Published {claim['published_at']} · Reviewed by {claim['reviewed_by']} "
                   f"· {claim['field']} characters {claim['start']}–{claim['end']}")

    st.subheader("Unassessed facts and outcome")
    st.write("Only the claim entries above have reviewer-supplied evidence. Other facts, "
             "company fundamentals, influencer identity, coordination and undisclosed "
             "payments have not been established by this audit. Future stock performance "
             "and whether a stock is a suitable purchase are not assessed.")
    for limitation in result.get("limitations", []):
        st.caption(limitation)
    st.download_button("Download audit JSON", json.dumps(result, indent=2, ensure_ascii=False,
                                                        allow_nan=False),
                       file_name="promotion-audit.json", mime="application/json")
else:
    st.info("Supply the promotion's own text, check the cutoff date and run the audit.")
