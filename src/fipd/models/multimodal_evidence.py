"""Multimodal evidence extraction and normalization across formats.

Provides shared, offline, network-free preprocessing for:
- Investment promotional graphics
- Email screenshots
- SMS screenshots

Includes:
- Robust OCR text normalization with OCR-artifact repair
- Local QR code extraction without network access
- Safe visible URL and domain parsing
- Content format detection (investment, email, SMS)
- Objective evidence indicator extraction (scam cues, legitimate cues, simulation markers)
- Review state determination when evidence is missing or ambiguous
"""
from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlparse

import cv2
import numpy as np
from PIL import Image

URL_REGEX = re.compile(
    r"(?:https?[:：]?(?:\/\/|ःll|ll|Il|\s*//)?|www\.)\S+|\b[a-z0-9][a-z0-9.-]+\.(?:com|net|org|co|in|uk|io|ly|info|xyz|top|net)(?:/\S*)?",
    re.I,
)

# Common OCR misreadings of URLs
OCR_URL_FIXES = [
    (re.compile(r"https?[\s:：]+(?:Il|ll|//|/)\s*", re.I), "http://"),
    (re.compile(r"(?<!:)//\s*", re.I), "://"),
]


def clean_ocr_text(text: str) -> str:
    """Repair common OCR artifacts before downstream processing."""
    if not isinstance(text, str):
        return ""
    result = text
    for pattern, repl in OCR_URL_FIXES:
        result = pattern.sub(repl, result)
    return result.strip()


def normalize_multimodal_text(text: str) -> str:
    """Remove exact URLs and domain names so models learn content patterns, not specific hostnames."""
    cleaned = clean_ocr_text(text)
    masked = URL_REGEX.sub(" URL ", cleaned.lower())
    # Standardize whitespace
    return re.sub(r"\s+", " ", masked).strip()


def decode_qr(image: Image.Image) -> list[str]:
    """Locally decode QR payloads without network access or following links."""
    try:
        # Convert PIL to BGR OpenCV format
        rgb = np.asarray(image.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        detector = cv2.QRCodeDetector()
        payload, points, _ = detector.detectAndDecode(bgr)
        if payload and payload.strip():
            return [payload.strip()[:2048]]
        # In case points were detected but single-decode missed, try multi-detect
        retval, decoded_info, _, _ = detector.detectAndDecodeMulti(bgr)
        if retval and decoded_info:
            return [p.strip()[:2048] for p in decoded_info if p and p.strip()]
        return []
    except Exception:
        return []


def extract_visible_urls(text: str) -> list[str]:
    """Extract visible URLs from text without performing network resolution."""
    cleaned = clean_ocr_text(text)
    matches = URL_REGEX.findall(cleaned)
    urls = []
    for m in matches:
        m = m.strip(".,;:()[]{}<>\"'")
        if m and m not in urls:
            urls.append(m)
    return urls


def extract_domains(text: str) -> list[str]:
    """Extract domain names from visible text."""
    urls = extract_visible_urls(text)
    domains = set()
    for u in urls:
        if not u.startswith(("http://", "https://")):
            u = "http://" + u
        try:
            parsed = urlparse(u)
            if parsed.hostname:
                domains.add(parsed.hostname.lower())
        except Exception:
            continue
    return sorted(domains)


def detect_format(image_width: int, image_height: int, text: str, qr_payloads: list[str] | None = None) -> str:
    """Classify the content format based on visual layout and textual markers.

    Returns:
        "email_screenshot" | "sms_screenshot" | "investment_graphic"
    """
    lower = text.lower()
    aspect_ratio = image_height / max(1, image_width)

    # Email markers: header fields, recipient/sender lines, typical email client headers
    email_markers = [
        r"\bfrom\s*:\s*\S+@",
        r"\bto\s*:\s*\S+",
        r"\bsubject\s*:",
        r"\bexternal\s+email\b",
        r"\bunsubscribe\b",
        r"\bregards\s*,\s*\n",
        r"\b(?:reply|forward|unread|inbox)\b",
        r"\b(?:caniphish|policy)\b",
    ]
    email_hits = sum(1 for m in email_markers if re.search(m, lower))

    # SMS markers: phone numbers, short chat status, chat layout, cellular status bar
    sms_markers = [
        r"\bmessages\b",
        r"\bsms/mms\b",
        r"\bdelivered\b",
        r"\btoday\s+\d{1,2}[:.]\d{2}",
        r"\byesterday\s+\d{1,2}[:.]\d{2}",
        r"\b\+\d{1,3}\s*\(\d{3}\)",
        r"\breply\s+(?:stop|yes|no|\d)\b",
        r"\b(?:5g|4g|lte)\b",
        r"\bclick\s+here\s+to\s+claim\b",
    ]
    sms_hits = sum(1 for m in sms_markers if re.search(m, lower))

    # Investment markers: market terms, returns, NAV, mutual funds
    investment_markers = [
        r"\binvest(?:ment|or|ing)?\b",
        r"\bmutual\s+fund\b",
        r"\bportfolio\b",
        r"\bshares?\b",
        r"\bstocks?\b",
        r"\bnav\b",
        r"\bdividend\b",
        r"\bcapital\b",
        r"\bsecurities\b",
        r"\bsebi\b",
        r"\bamfi\b",
    ]
    inv_hits = sum(1 for m in investment_markers if re.search(m, lower))

    has_email_header = bool(re.search(r"\b(?:from\s*:\s*\S+@|subject\s*:|to\s*:\s*\S+|external\s+email\b|unsubscribe\b)", lower) or "@" in text)
    if inv_hits >= 2 and not has_email_header:
        return "investment_graphic"

    if has_email_header or email_hits >= 2:
        return "email_screenshot"

    if sms_hits >= 2 or (sms_hits >= 1 and len(text) < 300):
        return "sms_screenshot"

    if inv_hits >= 1:
        return "investment_graphic"

    # Default by aspect ratio and structure
    if aspect_ratio >= 1.25 and len(text) < 400:
        return "sms_screenshot"
    if "@" in text or "reply" in lower or "unread" in lower:
        return "email_screenshot"

    return "investment_graphic"


def extract_evidence_indicators(text: str, qr_payloads: list[str] | None = None) -> dict[str, Any]:
    """Identify objective security evidence and determine whether manual review is required."""
    qr_payloads = qr_payloads or []
    cleaned = clean_ocr_text(text)
    lower = cleaned.lower()

    indicators = []
    phishing_cues = []
    scam_cues = []
    legitimate_cues = []
    review_reasons = []

    # 1. Check QR simulation and content
    is_simulation = False
    for qr in qr_payloads:
        if "phishing" in qr.lower() or "simulation" in qr.lower() or "real campaign" in qr.lower():
            is_simulation = True
            review_reasons.append("simulated_phishing_qr")
            indicators.append("QR payload contains educational phishing simulation text.")
        elif qr.startswith("otpauth://"):
            legitimate_cues.append("totp_authenticator_qr")
            indicators.append("QR payload contains standard RFC 6238 TOTP authenticator URI.")
        else:
            indicators.append(f"QR payload decoded locally: {qr[:60]}...")

    # 2. Deceptive / Phishing cues
    if re.search(r"\b(?:congratulations|winner|won|claim\s+your|gift\s*card|free\s+iphone|\$500|\$250|\$350)\b", lower):
        phishing_cues.append("reward_incentive_lure")
        indicators.append("Message offers reward, gift card, or prize incentive.")

    if re.search(r"\b(?:password|credential|pin|ssn|deactivated|suspended|locked|unauthorized|unusual\s+activity)\b", lower):
        phishing_cues.append("urgency_or_credential_prompt")
        indicators.append("Message uses urgent coercion or prompts for security credentials.")

    if re.search(r"\b(?:guaranteed|risk-free|100%\s+guarantee|double\s+your|no\s+risk|secret\s+signal|never\s+loss)\b", lower):
        scam_cues.append("unrealistic_guarantee")
        indicators.append("Claim promises guaranteed investment returns or zero-risk profits.")

    # 3. Legitimate / Compliance cues
    if re.search(r"\b(?:mutual\s+fund\s+investments\s+are\s+subject\s+to\s+market\s+risks|read\s+all\s+scheme\s+related\s+documents)\b", lower):
        legitimate_cues.append("statutory_sebi_amfi_disclaimer")
        indicators.append("Contains standard regulatory mutual fund risk disclosure.")

    if re.search(r"\b(?:will\s+never\s+ask\s+(?:you\s+)?for|do\s+not\s+share\s+or\s+disclose|beware\s+of\s+frauds)\b", lower):
        legitimate_cues.append("explicit_anti_fraud_warning")
        indicators.append("Contains explicit banking/identity anti-fraud guidance.")

    if re.search(r"\b(?:intranet\.\S+|sso\.\S+|workday\.\S+)\b", lower):
        legitimate_cues.append("internal_enterprise_domain")
        indicators.append("References internal corporate intranet or SSO portal.")

    # 4. Review determination
    needs_review = False
    if is_simulation:
        needs_review = True
    elif len(cleaned.strip()) < 15 and not qr_payloads:
        needs_review = True
        review_reasons.append("insufficient_ocr_text")
    elif phishing_cues and legitimate_cues:
        needs_review = True
        review_reasons.append("conflicting_evidence")

    return {
        "indicators": indicators,
        "phishing_cues": phishing_cues,
        "scam_cues": scam_cues,
        "legitimate_cues": legitimate_cues,
        "is_simulation": is_simulation,
        "needs_review": needs_review,
        "review_reasons": review_reasons,
        "visible_urls": extract_visible_urls(cleaned),
        "domains": extract_domains(cleaned),
    }


class PlattCalibrator:
    """Calibrates raw decision margins into posterior probabilities using validation data."""

    def __init__(self):
        from sklearn.linear_model import LogisticRegression
        self.lr = LogisticRegression(C=1.0, solver="lbfgs")
        self.fitted = False

    def fit(self, margins: np.ndarray, labels: np.ndarray):
        margins = np.asarray(margins).reshape(-1, 1)
        labels = np.asarray(labels, dtype=int)
        self.lr.fit(margins, labels)
        self.fitted = True
        return self

    def predict_proba(self, margins: np.ndarray) -> np.ndarray:
        if not self.fitted:
            s = 1.0 / (1.0 + np.exp(-np.asarray(margins)))
            return np.column_stack([1.0 - s, s])
        margins = np.asarray(margins).reshape(-1, 1)
        return self.lr.predict_proba(margins)
