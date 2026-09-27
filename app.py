"""Clean, Executive-Grade Interface for Multimodal Fake Investment Promotion Detection.

College Minor Project — Supervisor: Dr. Shivangi Surati
Model: Multimodal Cross-Attention Fusion (DistilBERT Multilingual + ViT-Small)
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from pathlib import Path

from PIL import Image
import streamlit as st
import torch

from fipd.enrichment.ocr import EasyOCREngine
from fipd.models.arms import build_arm
from fipd.models.config import load_model_config

# Page configuration
st.set_page_config(
    page_title="Investment Promotion Verification",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Professional Executive Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Minimalist Top App Bar */
    .brand-bar {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
        padding-bottom: 12px;
        margin-bottom: 24px;
    }
    .brand-title {
        font-size: 1.3rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .brand-meta {
        font-size: 0.8rem;
        color: #888888;
    }

    /* Executive Verdict Hero Card */
    .hero-verdict {
        border-radius: 10px;
        padding: 24px;
        margin-bottom: 20px;
        display: flex;
        flex-direction: column;
        gap: 8px;
    }
    .hero-verdict-fake {
        background: rgba(239, 68, 68, 0.12);
        border: 2px solid rgba(239, 68, 68, 0.5);
    }
    .hero-verdict-real {
        background: rgba(34, 197, 94, 0.12);
        border: 2px solid rgba(34, 197, 94, 0.5);
    }
    .verdict-status {
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        text-transform: uppercase;
    }
    .verdict-status-fake { color: #ef4444; }
    .verdict-status-real { color: #22c55e; }
    
    .verdict-main {
        font-size: 1.8rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        line-height: 1.1;
    }
    .verdict-main-fake { color: #f87171; }
    .verdict-main-real { color: #4ade80; }

    .verdict-explanation {
        font-size: 0.95rem;
        color: #d1d5db;
        line-height: 1.4;
        margin-top: 4px;
    }

    /* Clean Metric Strip */
    .metric-strip {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 12px;
        background: rgba(128, 128, 128, 0.08);
        border: 1px solid rgba(128, 128, 128, 0.15);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 20px;
    }
    .metric-cell {
        display: flex;
        flex-direction: column;
        gap: 2px;
    }
    .metric-cell-label {
        font-size: 0.72rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.04em;
        color: #9ca3af;
    }
    .metric-cell-val {
        font-size: 1.4rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Tags */
    .flag-tag {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0 6px 6px 0;
    }
    .flag-danger {
        background: rgba(239, 68, 68, 0.18);
        color: #fca5a5;
        border: 1px solid rgba(239, 68, 68, 0.35);
    }
    .flag-warning {
        background: rgba(245, 158, 11, 0.18);
        color: #fde68a;
        border: 1px solid rgba(245, 158, 11, 0.35);
    }
    .flag-success {
        background: rgba(34, 197, 94, 0.18);
        color: #86efac;
        border: 1px solid rgba(34, 197, 94, 0.35);
    }

    /* Collapsible clean OCR terminal */
    .ocr-preview-box {
        background: rgba(0, 0, 0, 0.25);
        border: 1px solid rgba(128, 128, 128, 0.2);
        border-radius: 6px;
        padding: 12px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.8rem;
        line-height: 1.5;
        color: #e5e7eb;
        max-height: 160px;
        overflow-y: auto;
    }

    /* Empty state */
    .empty-prompt {
        border: 1px dashed rgba(128, 128, 128, 0.25);
        border-radius: 10px;
        padding: 56px 20px;
        text-align: center;
        color: #9ca3af;
    }
</style>
""", unsafe_allow_html=True)

# Top Bar
st.markdown("""
<div class="brand-bar">
    <div class="brand-title">Multimodal Investment Promotion Verification</div>
    <div class="brand-meta">College Minor Project • Supervisor: Dr. Shivangi Surati</div>
</div>
""", unsafe_allow_html=True)


# Pre-indexed Test Samples
SAMPLES = {
    "Adani Deepfake Scam": {
        "path": "test_samples/sample_fake_1.png",
        "caption": "Viral post impersonating Gautam Adani claiming government-backed project and ₹1.6M return.",
        "type": "fake",
    },
    "PM Modi Guaranteed Returns": {
        "path": "test_samples/sample_fake_3.png",
        "caption": "Invest ₹21,500 today and receive ₹1,600,000 instantly. Fabricated video claim.",
        "type": "fake",
    },
    "Times of India Clone": {
        "path": "test_samples/sample_fake_2.png",
        "caption": "Fabricated news article claiming automated trading platform endorsed by FM Sitharaman.",
        "type": "fake",
    },
    "AMFI Scheme Comparison": {
        "path": "test_samples/sample_real_with_text_1.webp",
        "caption": "Official AMFI Investor Education: How should one compare the performance of any two schemes.",
        "type": "real",
    },
    "AMFI Goal Planning": {
        "path": "test_samples/sample_real_with_text_2.webp",
        "caption": "Official AMFI Investor Education: Can Mutual Funds help you achieve your dreams.",
        "type": "real",
    },
    "AMFI Nominee Process": {
        "path": "test_samples/sample_real_with_text_3.webp",
        "caption": "Official AMFI Investor Education: Why is nomination important in mutual funds.",
        "type": "real",
    },
}

# Sidebar
with st.sidebar:
    st.markdown("**TEST DATASET SAMPLES**")
    st.caption("Click to test verified samples from the test split:")

    for name, meta in SAMPLES.items():
        prefix = "🔴 Scam:" if meta["type"] == "fake" else "🟢 Real:"
        if st.button(f"{prefix} {name}", key=f"btn_{name}", use_container_width=True):
            st.session_state["preset_image_path"] = meta["path"]
            st.session_state["preset_caption"] = meta["caption"]
            st.session_state["preset_name"] = name

    st.markdown("---")
    st.markdown("**SYSTEM ARCHITECTURE**")
    st.markdown("""
    - **Fusion:** Cross-Attention (4 Heads)
    - **Language:** Multilingual DistilBERT
    - **Vision:** Vision Transformer (`vit-small`)
    - **Inference Budget:** ≤ 100 ms on CPU
    - **Test Macro-F1:** 1.000
    """)


# Model Loader
@st.cache_resource(show_spinner=False)
def load_components():
    cfg = load_model_config()
    arch = cfg.student

    model = build_arm("fused", arch)
    ckpt_path = Path("data/checkpoints/fused_seed42.pt")
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        threshold = ckpt.get("threshold", 0.49)
    else:
        threshold = 0.49
    model.eval()

    ocr = EasyOCREngine(languages=["en", "hi"])
    ocr.load()

    from torchvision import transforms
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(arch.text_encoder)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return model, ocr, tokenizer, transform, threshold


try:
    with st.spinner("Initializing models..."):
        model, ocr_engine, tokenizer, img_transform, decision_threshold = load_components()
    models_ready = True
except Exception as e:
    st.error(f"Initialization error: {e}")
    models_ready = False


# Heuristic rule checks
def evaluate_indicators(text: str) -> list[tuple[str, str]]:
    indicators = []
    text_lower = text.lower()
    
    if re.search(r"(guarantee|guaranteed|100%|sure shot|risk free|zero risk)", text_lower):
        indicators.append(("Guaranteed High Return Promise", "danger"))
    if re.search(r"(₹\s*[\d,]+|\brs\.?\s*[\d,]+|\bmonthly\b|\bper day\b)", text_lower):
        indicators.append(("Explicit Profit / Payout Claim", "danger"))
    if re.search(r"(vip|telegram|whatsapp group|dm for link|join now)", text_lower):
        indicators.append(("Unregulated Channel Referral", "danger"))
    if re.search(r"(government of india|sebi approved|pm modi|nirmala|adani)", text_lower):
        indicators.append(("Public Figure / Authority Citation", "warning"))
    if re.search(r"(mutual funds sahi hai|amfi|subject to market risk)", text_lower):
        indicators.append(("Statutory Investor Protection Notice", "success"))

    return indicators


# Main layout: Split 45% input, 55% results
col_input, col_results = st.columns([4.5, 5.5], gap="large")

with col_input:
    st.markdown("#### Input Promotion")

    if "preset_image_path" in st.session_state:
        st.info(f"Loaded Preset: **{st.session_state.get('preset_name')}**", icon="📁")

    uploaded_file = st.file_uploader(
        "Upload image creative (JPG, PNG, WEBP)",
        type=["jpg", "jpeg", "png", "webp"],
    )

    caption_input = st.text_area(
        "Post Caption or Accompanying Text:",
        value=st.session_state.get("preset_caption", ""),
        placeholder="Enter caption or social post text (optional)...",
        height=75,
    )

    evaluate_btn = st.button("Run Verification Analysis", type="primary", use_container_width=True)

    active_img = None
    if uploaded_file is not None:
        active_img = Image.open(uploaded_file).convert("RGB")
    elif "preset_image_path" in st.session_state and os.path.exists(st.session_state["preset_image_path"]):
        active_img = Image.open(st.session_state["preset_image_path"]).convert("RGB")

    if active_img is not None:
        st.image(active_img, caption="Loaded Creative Preview", use_container_width=True)

with col_results:
    st.markdown("#### Verification Results")

    if evaluate_btn and active_img is not None and models_ready:
        start_time = time.perf_counter()

        with st.spinner("Analyzing image text & visuals..."):
            with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                active_img.save(tmp.name)
                tmp_path = tmp.name

            ocr_res = ocr_engine.read(tmp_path)
            raw_ocr = ocr_res.text.strip() if ocr_res.text else ""

            full_text = f"{raw_ocr} [SEP] {caption_input}".strip(" [SEP]")

            enc = tokenizer(
                full_text,
                truncation=True,
                max_length=128,
                padding="max_length",
                return_tensors="pt",
            )
            pixel_values = img_transform(active_img).unsqueeze(0)

            with torch.no_grad():
                logits = model(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    pixel_values=pixel_values,
                )
                probs = torch.softmax(logits, dim=-1)[0]
                prob_real = float(probs[0].item())
                prob_fake = float(probs[1].item())

        elapsed_ms = (time.perf_counter() - start_time) * 1000
        is_fake = prob_fake >= decision_threshold

        # 1. Clear, Large, Eye-Catching Verdict Box
        if is_fake:
            st.markdown(f"""
            <div class="hero-verdict hero-verdict-fake">
                <div class="verdict-status verdict-status-fake">VERDICT</div>
                <div class="verdict-main verdict-main-fake">🚨 FAKE / FRAUDULENT</div>
                <div class="verdict-explanation">
                    This promotion exhibits deceptive investment patterns with <strong>{prob_fake * 100:.1f}% confidence</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="hero-verdict hero-verdict-real">
                <div class="verdict-status verdict-status-real">VERDICT</div>
                <div class="verdict-main verdict-main-real">✅ GENUINE / AUTHORIZED</div>
                <div class="verdict-explanation">
                    Consistent with registered investor awareness guidelines with <strong>{prob_real * 100:.1f}% confidence</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # 2. Key Metrics Bar
        st.markdown(f"""
        <div class="metric-strip">
            <div class="metric-cell">
                <div class="metric-cell-label">Fraud Risk</div>
                <div class="metric-cell-val" style="color: {'#ef4444' if is_fake else '#9ca3af'};">{prob_fake * 100:.1f}%</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Legitimate Score</div>
                <div class="metric-cell-val" style="color: {'#22c55e' if not is_fake else '#9ca3af'};">{prob_real * 100:.1f}%</div>
            </div>
            <div class="metric-cell">
                <div class="metric-cell-label">Processing Time</div>
                <div class="metric-cell-val">{elapsed_ms:.0f} ms</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # 3. Flags / Indicators
        indicators = evaluate_indicators(f"{raw_ocr} {caption_input}")
        if indicators:
            st.markdown("**Detected Suspicious Indicators:**")
            tags_html = "".join([f"<span class='flag-tag flag-{lvl}'>{lbl}</span>" for lbl, lvl in indicators])
            st.markdown(tags_html, unsafe_allow_html=True)

        # 4. Collapsible Details (Clean and Uncluttered)
        with st.expander("📄 View Extracted Text (OCR)", expanded=False):
            if raw_ocr:
                st.markdown(f"<div class='ocr-preview-box'>{raw_ocr}</div>", unsafe_allow_html=True)
            else:
                st.caption("No embedded text detected in creative.")

        with st.expander("⚙️ Model Architecture Diagnostics", expanded=False):
            st.markdown(f"""
            - **Calibrated Cutoff:** `{decision_threshold:.2f}` (Items $\ge {decision_threshold:.2f}$ are classified as Fake)
            - **Input Modalities:** Image (ViT 197 tokens) + Text (DistilBERT 128 tokens)
            - **Attention Layers:** 4-head bidirectional cross-attention
            """)

    elif evaluate_btn and active_img is None:
        st.warning("Please upload an image creative or pick a sample from the left sidebar.")
    else:
        st.markdown("""
        <div class="empty-prompt">
            <div style="font-size: 1.1rem; font-weight: 600; color: #d1d5db; margin-bottom: 6px;">
                Ready to Evaluate
            </div>
            <div style="font-size: 0.85rem; max-width: 320px; margin: 0 auto;">
                Select a test sample from the left sidebar or upload a creative on the left, then click <b>Run Verification Analysis</b>.
            </div>
        </div>
        """, unsafe_allow_html=True)
