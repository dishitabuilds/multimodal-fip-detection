"""Tests for multimodal cross-format evidence extraction, dataset integrity, and models."""
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.models.multimodal_evidence import (
    clean_ocr_text, decode_qr, detect_format,
    extract_domains, extract_evidence_indicators, extract_visible_urls,
    normalize_multimodal_text,
)


def test_clean_ocr_text_should_repair_common_url_artifacts():
    broken = "Claim yours at http Iltarget.com/promo and verify at https ://example.com/test"
    cleaned = clean_ocr_text(broken)
    assert "http://target.com/promo" in cleaned
    assert "http://example.com/test" in cleaned


def test_normalize_multimodal_text_should_mask_exact_urls():
    text = "Visit https://bank-login-update.com/account or www.scam-bonus.org for $500 reward"
    normalized = normalize_multimodal_text(text)
    assert "https://bank-login-update.com" not in normalized
    assert "www.scam-bonus.org" not in normalized
    assert "URL" in normalized
    assert "500 reward" in normalized


def test_detect_format_should_distinguish_email_sms_and_investment():
    # Email layout
    email_text = ("Subject: Policy Update Notice\nFrom: hr@nebulon.com\nTo: team@nebulon.com\n"
                  "Please review the attached workplace policy.\nRegards,\nCompliance Team")
    assert detect_format(760, 520, email_text) == "email_screenshot"

    # SMS layout (tall mobile view)
    sms_text = "Target Alerts\nToday 10.14 AM\nCongratulations! You have won a $500 Target gift card.\nDelivered"
    assert detect_format(400, 680, sms_text) == "sms_screenshot"

    # Investment promotional graphic (square card / market terms)
    inv_text = ("Monthly SIP Wealth\nStart your mutual fund portfolio.\n"
                "Mutual fund investments are subject to market risks. Read all scheme related documents.")
    assert detect_format(600, 600, inv_text) == "investment_graphic"


def test_decode_qr_should_read_local_payloads_offline():
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    payload = "In a real campaign, this would lead to a phishing website."
    enc = cv2.QRCodeEncoder.create()
    mat = enc.encode(payload)
    scaled = cv2.resize(mat, (120, 120), interpolation=cv2.INTER_NEAREST)
    img = Image.fromarray(np.stack([scaled] * 3, axis=-1))

    decoded = decode_qr(img)
    assert decoded == [payload]


def test_extract_evidence_indicators_should_flag_simulations_and_disclaimers():
    # Simulated phishing email
    sim_ev = extract_evidence_indicators("Review policy and win iPhone",
                                         qr_payloads=["In a real campaign, this would lead to a phishing website."])
    assert sim_ev["is_simulation"] is True
    assert sim_ev["needs_review"] is True
    assert "simulated_phishing_qr" in sim_ev["review_reasons"]

    # Statutory disclaimer
    legit_ev = extract_evidence_indicators("Mutual fund investments are subject to market risks.")
    assert "statutory_sebi_amfi_disclaimer" in legit_ev["legitimate_cues"]
    assert legit_ev["needs_review"] is False

    # Banking alert with anti-fraud warning
    bank_ev = extract_evidence_indicators("Enter OTP 4920. Do NOT share or disclose URN to anyone. Beware of frauds.")
    assert "explicit_anti_fraud_warning" in bank_ev["legitimate_cues"]


def test_multimodal_dataset_and_split_integrity():
    dataset_dir = ROOT / "data/interim/multimodal-generalization"
    if not (dataset_dir / "records.jsonl").is_file():
        pytest.skip("Dataset has not been generated yet.")

    records = [json.loads(line) for line in (dataset_dir / "records.jsonl").read_text().splitlines()]
    manifest = json.loads((dataset_dir / "split_manifest.json").read_text())

    assert len(records) >= 100
    assert manifest["frozen_test_manifest"] is True

    # Check balance across formats
    formats = {r["format"] for r in records}
    assert formats == {"investment_graphic", "email_screenshot", "sms_screenshot"}

    for fmt in formats:
        fmt_recs = [r for r in records if r["format"] == fmt]
        fakes = sum(1 for r in fmt_recs if r["label"] == "fake")
        reals = sum(1 for r in fmt_recs if r["label"] == "real")
        assert fakes > 0 and reals > 0
        assert fakes == reals, f"Format {fmt} should have balanced classes"

    # Verify no campaign leaks across splits
    rec_by_uid = {r["uid"]: r for r in records}
    train_camps = {rec_by_uid[uid]["campaign_id"] for uid in manifest["splits"]["train"]}
    val_camps = {rec_by_uid[uid]["campaign_id"] for uid in manifest["splits"]["validation"]}
    test_camps = {rec_by_uid[uid]["campaign_id"] for uid in manifest["splits"]["test"]}

    assert not (train_camps & val_camps), "Leak between train and val campaigns!"
    assert not (train_camps & test_camps), "Leak between train and test campaigns!"
    assert not (val_camps & test_camps), "Leak between val and test campaigns!"


def test_trained_multimodal_model_artifact_and_calibration():
    artifact_path = ROOT / "data/processed/multimodal_generalization/model.joblib"
    if not artifact_path.is_file():
        pytest.skip("Multimodal generalization artifact not found.")

    import joblib
    artifact = joblib.load(artifact_path)
    assert artifact["model_version"] == "multimodal_generalization_v1"
    specialists = artifact["models"]["specialists"]
    assert set(specialists.keys()) == {"investment_graphic", "email_screenshot", "sms_screenshot"}

    # Calibrators produce valid probabilities
    for fmt, sp in specialists.items():
        margins = np.array([-1.5, -0.2, 0.0, 0.3, 1.5])
        probs = sp["calibrator"].predict_proba(margins)[:, 1]
        assert (probs >= 0.0).all() and (probs <= 1.0).all()
        # Monotonically increasing with decision margin
        assert (np.diff(probs) >= 0.0).all()


def test_multimodal_mode_runtime_and_classifier_loading():
    from fipd.frontend import analysis

    runtime = analysis.mode_runtime(ROOT, "multimodal")
    assert runtime["mode"] == "multimodal"
    assert runtime["purpose"] == "cross_format_specialist"
    assert runtime["checkpoint_available"] is True
    assert runtime["experiment_id"] == "multimodal_generalization_v1"

    components = analysis.load_classifier(ROOT, "multimodal")
    assert components.mode == "multimodal"
    assert components.purpose == "cross_format_specialist"
    assert hasattr(components.encoder, "encode_images")
    assert "specialists" in components.artifact["models"]


def test_multimodal_review_band_includes_both_edges():
    from fipd.frontend.analysis import multimodal_decision

    assert multimodal_decision(0.4499) == ("real", None)
    assert multimodal_decision(0.45) == (None, "uncertain_prediction")
    assert multimodal_decision(0.50) == (None, "uncertain_prediction")
    assert multimodal_decision(0.55) == (None, "uncertain_prediction")
    assert multimodal_decision(0.5501) == ("fake", None)
    with pytest.raises(ValueError):
        multimodal_decision(float("nan"))


def test_multimodal_analyze_image_on_development_challenge_images():
    from fipd.frontend import analysis

    challenge_dir = ROOT / "data/interim/generalization-challenge"
    if not (challenge_dir / "phishing_email_qr.png").is_file():
        pytest.skip("Challenge images not found.")

    # 1. Simulated Phishing Email with QR
    with Image.open(challenge_dir / "phishing_email_qr.png") as img:
        res1 = analysis.analyze_image(
            img,
            caption="Review our remote working policy and win an iPhone",
            mode="multimodal",
            root=ROOT,
        )
    assert res1["status"] == "complete"
    assert res1["mode"] == "multimodal"
    assert res1["detected_format"] == "email_screenshot"
    assert res1["label"] in {"fake", "real", None}
    assert len(res1["qr_payloads"]) > 0
    assert "phishing website" in res1["qr_payloads"][0].lower()

    # 2. Smishing reward image
    with Image.open(challenge_dir / "smishing_reward.png") as img:
        res2 = analysis.analyze_image(
            img,
            caption="",
            mode="multimodal",
            root=ROOT,
        )
    assert res2["status"] == "complete"
    assert res2["mode"] == "multimodal"
    assert res2["detected_format"] == "sms_screenshot"
    assert res2["review_reason"] == "uncertain_prediction"
    assert res2["label"] is None
    assert 0.45 <= res2["class_scores"]["fake"] <= 0.55
