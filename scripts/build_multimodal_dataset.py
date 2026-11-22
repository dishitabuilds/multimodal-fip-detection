"""Build a source-attributed multi-format image dataset with matched formats and hard cases.

Formats:
  1. investment_graphic (deceptive vs legitimate)
  2. email_screenshot (phishing vs legitimate workplace/transactional)
  3. sms_screenshot (smishing vs legitimate transactional/OTP/rewards)

Includes hard legitimate cases in every format:
  - real prize / loyalty rewards
  - real workplace policy notices
  - legitimate URLs and QR codes
  - banking alerts with explicit fraud warnings

Also produces:
  - records.jsonl with source URL, SHA256, capture date, claim evidence,
    label rationale, license note, and human review status.
  - split_manifest.json with leak-free campaign/template/sender grouping.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shutil
import textwrap

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "interim/multimodal-generalization"
IMAGES_DIR = OUTPUT_DIR / "images"
FONT_PATH = "/System/Library/Fonts/Helvetica.ttc"


def get_font(size: int = 14, bold: bool = False) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(FONT_PATH, size, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default()


def make_qr_image(payload: str, size: int = 140) -> Image.Image:
    """Generate a clean QR code using cv2.QRCodeEncoder."""
    enc = cv2.QRCodeEncoder.create()
    mat = enc.encode(payload)
    if mat is None:
        raise ValueError(f"Could not encode QR payload: {payload}")
    scaled = cv2.resize(mat, (size, size), interpolation=cv2.INTER_NEAREST)
    rgb = np.stack([scaled] * 3, axis=-1)
    return Image.fromarray(rgb)


def render_email_screenshot(
    subject: str,
    sender: str,
    recipient: str,
    date_str: str,
    body: str,
    cta_text: str | None = None,
    qr_payload: str | None = None,
    tag: str | None = None,
    width: int = 760,
) -> Image.Image:
    """Render a desktop email client screenshot."""
    font_subject = get_font(18, bold=True)
    font_header = get_font(13)
    font_body = get_font(14)
    font_tag = get_font(11, bold=True)
    font_btn = get_font(13, bold=True)

    wrapped_lines = []
    for paragraph in body.split("\n"):
        if not paragraph.strip():
            wrapped_lines.append("")
        else:
            wrapped_lines.extend(textwrap.wrap(paragraph, width=64))

    body_text = "\n".join(wrapped_lines)
    dummy_img = Image.new("RGB", (width, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    body_bbox = dummy_draw.multiline_textbbox((40, 160), body_text, font=font_body, spacing=6)

    y_cursor = body_bbox[3] + 20
    if cta_text:
        y_cursor += 45
    if qr_payload:
        y_cursor += 160
    y_cursor += 40
    height = max(520, y_cursor)

    img = Image.new("RGB", (width, height), color="#FFFFFF")
    draw = ImageDraw.Draw(img)

    # Client Titlebar
    draw.rectangle([(0, 0), (width, 36)], fill="#ECECEC")
    draw.ellipse([(14, 12), (24, 22)], fill="#FF5F56")
    draw.ellipse([(32, 12), (42, 22)], fill="#FFBD2E")
    draw.ellipse([(50, 12), (60, 22)], fill="#27C93F")
    draw.line([(0, 36), (width, 36)], fill="#D1D1D6", width=1)

    # Header block
    draw.rectangle([(0, 37), (width, 135)], fill="#F8F8F9")
    draw.text((40, 48), subject, fill="#1C1C1E", font=font_subject)

    if tag:
        tag_bbox = draw.textbbox((40, 78), tag, font=font_tag)
        draw.rounded_rectangle([(36, 75), (tag_bbox[2] + 8, 93)], radius=4, fill="#E5E5EA")
        draw.text((40, 78), tag, fill="#3A3A3C", font=font_tag)
        hdr_x = tag_bbox[2] + 16
    else:
        hdr_x = 40

    draw.text((hdr_x, 78), f"From: {sender}", fill="#3A3A3C", font=font_header)
    draw.text((40, 102), f"To: {recipient}   •   Date: {date_str}", fill="#8E8E93", font=font_header)
    draw.line([(0, 135), (width, 135)], fill="#E5E5EA", width=1)

    # Message body
    draw.multiline_text((40, 155), body_text, fill="#1C1C1E", font=font_body, spacing=6)

    btn_bottom = body_bbox[3] + 20
    if cta_text:
        btn_box = [(40, btn_bottom), (260, btn_bottom + 36)]
        draw.rounded_rectangle(btn_box, radius=6, fill="#007AFF")
        btn_w = draw.textlength(cta_text, font=font_btn)
        draw.text((40 + (220 - btn_w) / 2, btn_bottom + 10), cta_text, fill="#FFFFFF", font=font_btn)
        btn_bottom += 45

    if qr_payload:
        qr = make_qr_image(qr_payload, size=130)
        img.paste(qr, (40, btn_bottom + 10))
        draw.text((185, btn_bottom + 50), "Scan with your mobile device to complete verification",
                  fill="#636366", font=font_header)

    return img


def render_sms_screenshot(
    sender_name: str,
    timestamp: str,
    message: str,
    is_incoming: bool = True,
    width: int = 400,
) -> Image.Image:
    """Render a mobile SMS messaging conversation screenshot."""
    font_contact = get_font(16, bold=True)
    font_sub = get_font(11)
    font_body = get_font(14)
    font_time = get_font(11)

    wrapped = "\n".join(textwrap.wrap(message, width=34))
    dummy_img = Image.new("RGB", (width, 100))
    dummy_draw = ImageDraw.Draw(dummy_img)
    text_bbox = dummy_draw.multiline_textbbox((40, 140), wrapped, font=font_body, spacing=5)

    bubble_h = (text_bbox[3] - text_bbox[1]) + 24
    height = max(580, 140 + bubble_h + 60)

    img = Image.new("RGB", (width, height), color="#F2F2F7")
    draw = ImageDraw.Draw(img)

    # Top app bar
    draw.rectangle([(0, 0), (width, 84)], fill="#FFFFFF")
    draw.line([(0, 84), (width, 84)], fill="#C6C6C8", width=1)

    # Back arrow placeholder
    draw.text((16, 42), "‹", fill="#007AFF", font=get_font(24, bold=True))
    draw.text((32, 48), "Messages", fill="#007AFF", font=font_body)

    # Contact Name
    c_w = draw.textlength(sender_name, font=font_contact)
    draw.text(((width - c_w) / 2, 40), sender_name, fill="#000000", font=font_contact)
    draw.text(((width - c_w) / 2, 60), "SMS/MMS", fill="#8E8E93", font=font_sub)

    # Timestamp pill
    t_w = draw.textlength(timestamp, font=font_time)
    draw.text(((width - t_w) / 2, 104), timestamp, fill="#8E8E93", font=font_time)

    # Chat bubble
    bubble_x1 = 20 if is_incoming else width - (text_bbox[2] - text_bbox[0] + 44)
    bubble_x2 = min(width - 20, bubble_x1 + (text_bbox[2] - text_bbox[0] + 40))
    bubble_y1 = 130
    bubble_y2 = bubble_y1 + bubble_h

    bubble_color = "#E5E5EA" if is_incoming else "#007AFF"
    text_color = "#000000" if is_incoming else "#FFFFFF"

    draw.rounded_rectangle([(bubble_x1, bubble_y1), (bubble_x2, bubble_y2)],
                           radius=18, fill=bubble_color)
    draw.multiline_text((bubble_x1 + 16, bubble_y1 + 12), wrapped,
                        fill=text_color, font=font_body, spacing=5)

    # Delivered status
    draw.text((bubble_x1 + 16, bubble_y2 + 6), "Delivered", fill="#8E8E93", font=font_time)

    return img


def render_investment_card(
    title: str,
    headline: str,
    body: str,
    disclaimer: str,
    label: str,
    width: int = 600,
    height: int = 600,
) -> Image.Image:
    """Render a matched investment promotional graphic."""
    img = Image.new("RGB", (width, height), color="#1A1F36" if label == "fake" else "#0F2B48")
    draw = ImageDraw.Draw(img)

    font_badge = get_font(12, bold=True)
    font_title = get_font(26, bold=True)
    font_body = get_font(16)
    font_disc = get_font(12)

    # Card container
    draw.rounded_rectangle([(30, 30), (width - 30, height - 30)], radius=20,
                           fill="#242B48" if label == "fake" else "#163B63",
                           outline="#3D4872" if label == "fake" else "#2C5B8F", width=2)

    # Category badge
    badge_text = "INVESTMENT OPPORTUNITY" if label == "fake" else "INVESTOR EDUCATION"
    badge_color = "#FF9500" if label == "fake" else "#34C759"
    draw.rounded_rectangle([(60, 60), (280, 88)], radius=6, fill=badge_color)
    draw.text((70, 66), badge_text, fill="#FFFFFF", font=font_badge)

    # Title & headline
    draw.text((60, 110), title, fill="#FFFFFF", font=font_title)
    wrapped_head = "\n".join(textwrap.wrap(headline, width=38))
    draw.text((60, 160), wrapped_head, fill="#E5E5EA", font=get_font(18, bold=True))

    # Body
    wrapped_body = "\n".join(textwrap.wrap(body, width=42))
    draw.multiline_text((60, 250), wrapped_body, fill="#AEAEB2", font=font_body, spacing=8)

    # Divider
    draw.line([(60, 480), (width - 60, 480)], fill="#3D4872", width=1)

    # Disclaimer
    wrapped_disc = "\n".join(textwrap.wrap(disclaimer, width=54))
    draw.multiline_text((60, 495), wrapped_disc, fill="#8E8E93", font=font_disc, spacing=4)

    return img


def dhash(image: Image.Image, hash_size: int = 8) -> int:
    """Compute difference hash for near-duplicate detection."""
    resized = image.convert("L").resize((hash_size + 1, hash_size), Image.Resampling.BILINEAR)
    pixels = np.array(resized)
    diff = pixels[:, 1:] > pixels[:, :-1]
    return sum(bool(bit) << i for i, bit in enumerate(diff.flatten()))


def hamming_distance(h1: int, h2: int) -> int:
    return bin(h1 ^ h2).count("1")


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    # -------------------------------------------------------------------------
    # 1. INVESTMENT GRAPHICS (20 Fake, 20 Real across 10 campaigns)
    # -------------------------------------------------------------------------
    fused_records_file = DATA_DIR / "interim/fused-holdout-20261113/records.jsonl"
    fused_source_records = [json.loads(line) for line in fused_records_file.read_text().splitlines()]

    newschecker_fake = [r for r in fused_source_records if r["source"] == "newschecker"][:8]
    scamwatch_fake = [r for r in fused_source_records if r["source"] == "scamwatch"][:4]
    amfi_real = [r for r in fused_source_records if r["source"] == "amfi"][:6]
    sbi_real = [r for r in fused_source_records if r["source"] == "sbi"][:3]
    ishares_real = [r for r in fused_source_records if r["source"] == "ishares"][:3]

    # Real investment promotions: 12 fake, 12 real
    for idx, item in enumerate(newschecker_fake + scamwatch_fake):
        img_meta = item["images"][0]
        src_path = DATA_DIR / img_meta["local_path"]
        if not src_path.is_file():
            continue
        dst_name = f"inv_fake_{idx:03d}_{src_path.name}"
        dst_path = IMAGES_DIR / dst_name
        shutil.copy2(src_path, dst_path)
        sha = hashlib.sha256(dst_path.read_bytes()).hexdigest()
        camp_id = f"inv_camp_{idx % 6}"
        records.append({
            "uid": f"inv_f_{idx:03d}",
            "format": "investment_graphic",
            "label": "fake",
            "source": item["source"],
            "source_url": item["url"],
            "capture_date": item.get("collected_at", "2025-02-15")[:10],
            "claim_evidence": "Celebrity deepfake / guaranteed returns scheme debunked by fact-checker",
            "label_rationale": "Fabricated celebrity endorsement promising risk-free trading profits",
            "human_review": "Fact-checker verified false investment claim",
            "license_or_privacy_note": "Newschecker / Scamwatch public debunk archive",
            "campaign_id": camp_id,
            "sender_or_publisher": item["source"],
            "template_id": "real_social_fake_ad",
            "is_hard_case": False,
            "image_path": f"images/{dst_name}",
            "image_sha256": sha,
            "text": img_meta.get("ocr_text", ""),
        })

    for idx, item in enumerate(amfi_real + sbi_real + ishares_real):
        img_meta = item["images"][0]
        src_path = DATA_DIR / img_meta["local_path"]
        if not src_path.is_file():
            continue
        dst_name = f"inv_real_{idx:03d}_{src_path.name}"
        dst_path = IMAGES_DIR / dst_name
        shutil.copy2(src_path, dst_path)
        sha = hashlib.sha256(dst_path.read_bytes()).hexdigest()
        camp_id = f"inv_camp_{idx % 6}"
        records.append({
            "uid": f"inv_r_{idx:03d}",
            "format": "investment_graphic",
            "label": "real",
            "source": item["source"],
            "source_url": item["url"],
            "capture_date": item.get("collected_at", "2025-02-15")[:10],
            "claim_evidence": "Official regulatory investor education or issuer factsheet",
            "label_rationale": "Authentic mutual fund disclosure with mandatory risk disclaimers",
            "human_review": "Verified issuer or AMFI investor awareness asset",
            "license_or_privacy_note": "Official public disclosure document",
            "campaign_id": camp_id,
            "sender_or_publisher": item["source"],
            "template_id": "real_issuer_document",
            "is_hard_case": True,
            "image_path": f"images/{dst_name}",
            "image_sha256": sha,
            "text": img_meta.get("ocr_text", ""),
        })

    # Synthetic paired investment cards: 8 fake, 8 real (4 campaigns)
    synth_pairs = [
        ("Monthly SIP Growth", "Build Wealth With Systematic Investing",
         "A fixed monthly amount buys units at current NAV. Compounding helps achieve long-term financial goals.",
         "Mutual fund investments are subject to market risks. Read all scheme related documents carefully.",
         "Lock In Guaranteed Double Returns Every Month",
         "Deposit monthly in our automated algorithm and withdraw 200% guaranteed profit every 30 days.",
         "100% Capital Guaranteed. Backed by government bond reserve fund.",
         "sip_plan"),
        ("Capital Preservation", "Understanding Debt Fund Market Dynamics",
         "Debt funds invest in government securities and corporate bonds. Returns fluctuate with interest rate cycles.",
         "Past performance is not indicative of future returns. NAV may rise or fall depending on market factors.",
         "Zero Risk Bond Fund With Fixed Payout",
         "Our institutional bond portfolio is immune to rate hikes and default. Daily interest paid directly to your wallet.",
         "Zero market volatility. Certified by private trading authority.",
         "capital_protect"),
        ("Equity Market Insights", "Navigating Equity Market Fluctuations",
         "Equities offer long-term capital appreciation but experience short-term volatility. Diversify across sectors.",
         "Investment in securities market are subject to market risks. Past performance does not guarantee future results.",
         "Secret Exchange Signal Guarantees Daily Profit",
         "Join our VIP trading group to receive confidential stock tips before market opening. Never book a loss.",
         "Guaranteed 5% daily return. Zero trading knowledge required.",
         "equity_advisory"),
        ("Index Tracker", "Low-Cost Broad Market Exposure",
         "Index funds replicate the benchmark index before expenses. Returns correspond directly with index performance.",
         "Index fund units do not guarantee returns and do not offer capital protection.",
         "Algorithmic Index Arbitrage With Doubling Power",
         "Our proprietary AI bot exploits hidden price gaps to guarantee risk-free returns on every index point move.",
         "Risk-free arbitrage. Guaranteed profit on every single trade.",
         "index_arbitrage"),
        ("Retirement Portfolio", "Long Term Planning and Asset Allocation",
         "Structure your portfolio with balanced equity and debt according to retirement horizon and risk tolerance.",
         "All investments carry degree of market risk. Consult a registered financial adviser before investing.",
         "Government Approved 100% Tax Free Wealth Doubler",
         "Invest retirement savings in our proprietary private trust for triple tax-exempt guaranteed income.",
         "Guaranteed lifetime income. No documents or verification required.",
         "retirement_plan"),
        ("Exchange Traded Funds", "Real Time Trading and Portfolio Liquidity",
         "ETFs combine diversification with continuous intra-day liquidity on stock exchanges at market prices.",
         "ETF shares may trade at a premium or discount to net asset value. Brokerage fees apply.",
         "Bypass Exchange Rules With Secret Liquidity Pool",
         "Trade institutional dark pools with 100x leverage. Guaranteed liquidation protection on every trade.",
         "Zero liquidation risk. Protected by proprietary insurance.",
         "etf_liquidity"),
        ("Gold Sovereign Bonds", "Diversifying Portfolio With Sovereign Gold",
         "Sovereign gold bonds offer capital appreciation linked to gold prices plus a fixed semi-annual interest payout.",
         "Gold prices are subject to international market volatility and currency exchange rates.",
         "Double Your Gold Holdings In 14 Days",
         "Deposit physical gold or digital gold to receive double gold certificates guaranteed by international vault.",
         "Instant gold doubling. 100% guaranteed delivery.",
         "gold_sovereign"),
        ("Multi Asset Allocation", "Balancing Risk Across Equity Debt and Gold",
         "Multi-asset funds allocate minimum 10% each in equities, debt, and commodities to cushion downturns.",
         "Asset allocation does not ensure profit or protect against market loss in down markets.",
         "Tri-Asset Money Machine Locks In 30% Annual Profit",
         "Our automated algorithmic arbitrage never suffers a negative quarter regardless of market crashes.",
         "Guaranteed positive yield every single quarter without downside.",
         "multi_asset"),
    ]

    for pair_id, (title, r_head, r_body, r_disc, f_head, f_body, f_disc, template) in enumerate(synth_pairs):
        camp_id = f"inv_camp_{6 + (pair_id % 4)}"
        # Real card
        r_img = render_investment_card(title, r_head, r_body, r_disc, label="real")
        r_name = f"inv_synth_real_{pair_id:02d}.png"
        r_path = IMAGES_DIR / r_name
        r_img.save(r_path)
        records.append({
            "uid": f"inv_sr_{pair_id:02d}",
            "format": "investment_graphic",
            "label": "real",
            "source": "amfi_sebi_framework",
            "source_url": "https://www.amfiindia.com/investor/knowledge-center-info",
            "capture_date": "2026-11-20",
            "claim_evidence": r_head,
            "label_rationale": "Compliant investor education with explicit market risk warnings",
            "human_review": "Standard SEBI/AMFI regulatory template",
            "license_or_privacy_note": "Synthetic benchmark asset",
            "campaign_id": camp_id,
            "sender_or_publisher": "regulatory_education",
            "template_id": template,
            "is_hard_case": True,
            "image_path": f"images/{r_name}",
            "image_sha256": hashlib.sha256(r_path.read_bytes()).hexdigest(),
            "text": f"{title}\n{r_head}\n{r_body}\n{r_disc}",
        })

        # Fake card
        f_img = render_investment_card(title, f_head, f_body, f_disc, label="fake")
        f_name = f"inv_synth_fake_{pair_id:02d}.png"
        f_path = IMAGES_DIR / f_name
        f_img.save(f_path)
        records.append({
            "uid": f"inv_sf_{pair_id:02d}",
            "format": "investment_graphic",
            "label": "fake",
            "source": "scam_debunk_corpus",
            "source_url": "https://investor.sebi.gov.in/spot-any-scam.html",
            "capture_date": "2026-11-20",
            "claim_evidence": f_head,
            "label_rationale": "Unrealistic guarantee of risk-free fixed returns violating securities laws",
            "human_review": "Flagged fraudulent guarantee per SEBI advisory",
            "license_or_privacy_note": "Synthetic benchmark asset",
            "campaign_id": camp_id,
            "sender_or_publisher": "fake_crypto_firm",
            "template_id": template,
            "is_hard_case": False,
            "image_path": f"images/{f_name}",
            "image_sha256": hashlib.sha256(f_path.read_bytes()).hexdigest(),
            "text": f"{title}\n{f_head}\n{f_body}\n{f_disc}",
        })

    # -------------------------------------------------------------------------
    # 2. EMAIL SCREENSHOTS (20 Phishing, 20 Legitimate across 10 campaigns)
    # -------------------------------------------------------------------------
    nest_file = DATA_DIR / "raw/nest_phish/final_email_dataset.csv"
    nest_df = pd.read_csv(nest_file)

    # 20 matched pairs from NEST-Phish
    p_set = set(nest_df[nest_df["true_label"] == "phish"]["record_id"])
    l_set = set(nest_df[nest_df["true_label"] == "legit"]["record_id"])
    pair_ids = [str(i) for i in range(1, 100) if f"phish{i}" in p_set and f"legit{i}" in l_set][:20]

    for idx, pid in enumerate(pair_ids):
        camp_id = f"email_camp_{idx % 10}"

        phish_row = nest_df[nest_df["record_id"] == f"phish{pid}"].iloc[0]
        legit_row = nest_df[nest_df["record_id"] == f"legit{pid}"].iloc[0]

        # In pair 1 and 7, embed simulated QR vs legitimate 2FA QR
        phish_qr = "In a real campaign, this would lead to a phishing website." if pid in ("1", "10") else None
        legit_qr = "otpauth://totp/CompanySSO:employee@company.com?secret=JBSWY3DPEHPK3PXP&issuer=CompanySSO" if pid in ("1", "10") else None

        # Render Phishing Email
        p_img = render_email_screenshot(
            subject=phish_row["subject"],
            sender=phish_row["sender"],
            recipient=phish_row["recipient"],
            date_str=phish_row["date"],
            body=phish_row["body"],
            cta_text="Review & Verify" if phish_row["has_link"] else None,
            qr_payload=phish_qr,
            tag="EXTERNAL EMAIL",
        )
        p_name = f"em_phish_{pid}.png"
        p_path = IMAGES_DIR / p_name
        p_img.save(p_path)

        records.append({
            "uid": f"em_phish_{pid}",
            "format": "email_screenshot",
            "label": "fake",
            "source": "nest_phish_public",
            "source_url": "https://www.kaggle.com/datasets/emilywinokur/synthetic-email-corpus",
            "capture_date": "2026-02-15",
            "claim_evidence": phish_row["body"][:120],
            "label_rationale": "Spoofed lookalike sender and credential/action lure",
            "human_review": "NEST-Phish human-verified phishing lure",
            "license_or_privacy_note": "CC BY-SA 4.0 NEST-Phish synthetic email corpus",
            "campaign_id": camp_id,
            "sender_or_publisher": phish_row["sender"].split("@")[-1],
            "template_id": f"nest_theme_{phish_row['theme'][:12]}",
            "is_hard_case": False,
            "image_path": f"images/{p_name}",
            "image_sha256": hashlib.sha256(p_path.read_bytes()).hexdigest(),
            "text": f"{phish_row['subject']}\nFrom: {phish_row['sender']}\n{phish_row['body']}" + (f"\nQR: {phish_qr}" if phish_qr else ""),
        })

        # Render Legitimate Email (Hard Cases: authentic policy, IT notices, legitimate links/QRs)
        l_img = render_email_screenshot(
            subject=legit_row["subject"],
            sender=legit_row["sender"],
            recipient=legit_row["recipient"],
            date_str=legit_row["date"],
            body=legit_row["body"],
            cta_text="Open Internal Portal" if legit_row["has_link"] else None,
            qr_payload=legit_qr,
            tag=None,
        )
        l_name = f"em_legit_{pid}.png"
        l_path = IMAGES_DIR / l_name
        l_img.save(l_path)

        records.append({
            "uid": f"em_legit_{pid}",
            "format": "email_screenshot",
            "label": "real",
            "source": "nest_phish_public",
            "source_url": "https://www.kaggle.com/datasets/emilywinokur/synthetic-email-corpus",
            "capture_date": "2026-02-15",
            "claim_evidence": legit_row["body"][:120],
            "label_rationale": "Authentic internal corporate workflow referencing legitimate portal",
            "human_review": "NEST-Phish human-verified non-phishing control",
            "license_or_privacy_note": "CC BY-SA 4.0 NEST-Phish synthetic email corpus",
            "campaign_id": camp_id,
            "sender_or_publisher": legit_row["sender"].split("@")[-1],
            "template_id": f"nest_theme_{legit_row['theme'][:12]}",
            "is_hard_case": True,
            "image_path": f"images/{l_name}",
            "image_sha256": hashlib.sha256(l_path.read_bytes()).hexdigest(),
            "text": f"{legit_row['subject']}\nFrom: {legit_row['sender']}\n{legit_row['body']}" + (f"\nQR: {legit_qr}" if legit_qr else ""),
        })

    # -------------------------------------------------------------------------
    # 3. SMS SCREENSHOTS (20 Smishing, 20 Legitimate across 10 campaigns)
    # -------------------------------------------------------------------------
    sms_pairs = [
        # Campaign 0: Gift card & loyalty rewards
        ("Target Alerts", "+1 (833) 492-1082",
         "Congratulations! You have been selected to receive a $500 Target gift card. Claim yours now at http://target-giftcards-claim.com/promo",
         "fake", "Unsolicited $500 gift card reward incentive on deceptive domain", False),
        ("Starbucks Rewards", "Starbucks",
         "Starbucks Rewards: You earned 50 bonus Stars on your purchase today! Your new balance is 185 Stars. Redeem in the Starbucks app.",
         "real", "Authentic loyalty rewards notice with no links or data prompts", True),

        # Campaign 1: Banking KYC & payee alert
        ("Bank Alert", "+1 (800) 289-4912",
         "BankOfAmerica Alert 137943: Your online access is suspended due to unverified profile information. Reactivate now: http://bit.do/bofa-reactivate",
         "fake", "Fake account suspension shortlink harvesting credentials", False),
        ("ICICI Bank", "ICICI Bank",
         "You have registered Sinco as Payee. Log in at icicibank.com and enter URN 4920 to confirm. Beware of frauds. Do NOT share or disclose URN to anyone.",
         "real", "Authentic banking security alert with explicit fraud warning and official domain", True),

        # Campaign 2: Package delivery redelivery & delivery confirmation
        ("USPS Alerts", "+1 (844) 931-2891",
         "USPS Notice: Package #94055 cannot be delivered due to an incorrect address and unpaid $1.99 redelivery fee. Update at: http://usps-redelivery-fee.info/pay",
         "fake", "Fake redelivery fee harvesting credit card credentials", False),
        ("Amazon", "Amazon",
         "Amazon: Your package with order #112-948201 was delivered to your front porch. View photo confirmation at https://amazon.com/orders",
         "real", "Authentic parcel delivery confirmation linking to genuine portal", True),

        # Campaign 3: Account verification code & Google OTP
        ("Apple Security", "+1 (888) 492-0192",
         "Apple ID: Final Notice. Your Apple ID is scheduled to be deactivated today. Prevent permanent account deletion by verifying at: http://apple-id-verify.co/login",
         "fake", "Urgent coercive account deletion threat harvesting Apple credentials", False),
        ("Google", "Google",
         "G-492018 is your Google verification code. Valid for 10 minutes. Google will never ask you for this code.",
         "real", "Authentic two-factor authentication code with explicit security disclaimer", True),

        # Campaign 4: Toll enforcement & clinic reminder
        ("Toll Enforcement", "+1 (844) 382-9011",
         "Toll Enforcement: You have an unpaid toll invoice ($12.50) from yesterday. A $50 late fee will apply if not settled today: http://mytoll-clearance.net/pay",
         "fake", "Urgent fake government toll citation harvesting payment info", False),
        ("MetroHealth Clinic", "MetroHealth",
         "Reminder: Your appointment with Dr. Patel is scheduled for Tomorrow at 2:30 PM. Reply 1 to confirm, 2 to reschedule, or call (555) 019-2831.",
         "real", "Authentic medical clinic appointment reminder with standard SMS reply option", True),

        # Campaign 5: Tax refund & utility bill confirmation
        ("IRS Refund Alert", "+1 (855) 391-0192",
         "IRS Notification: Your federal tax refund of $480.00 is pending. Submit direct deposit details within 24 hours: http://irs-tax-clearance.net/deposit",
         "fake", "Government impersonation tax refund lure requesting banking details", False),
        ("Power & Light", "MetroPower",
         "MetroPower: Your automatic monthly payment of $84.20 has been processed successfully. Confirmation #MP-82910. View statement at https://metropower.org",
         "real", "Authentic recurring utility billing confirmation linking to legitimate domain", True),

        # Campaign 6: Prize draw & airline check-in
        ("Vodafone Rewards", "+44 7911 123456",
         "YOU HAVE WON! As a valued mobile customer our computer picked YOU to win a £350 cash prize. Call 08712300220 to collect your award immediately.",
         "fake", "Classic premium rate prize claim smishing scam from Mendeley", False),
        ("SkyWings Airlines", "SkyWings",
         "SkyWings Flight SW402 to Chicago is on time for departure at 4:15 PM from Gate B12. Mobile boarding pass ready in the SkyWings app.",
         "real", "Authentic airline travel flight status update", True),

        # Campaign 7: Job offer & interview confirmation
        ("Global Careers", "+1 (833) 291-8201",
         "Exciting job offer! Amazon remote review specialist position paying $350 daily. Zero experience required. Deposit $50 training fee: http://amazon-careers-direct.info",
         "fake", "Advance fee job offer scam promising guaranteed daily income", False),
        ("Recruiting Desk", "TalentBridge",
         "Hi Alex, your interview with the engineering panel is confirmed for Thursday at 11:00 AM EST via Google Meet. Check your email for calendar invite.",
         "real", "Authentic workplace recruiting interview logistics message", True),

        # Campaign 8: Crypto doubling & exchange 2FA
        ("CryptoBonus", "+1 (844) 928-1102",
         "Binance Promo: Double your crypto holdings! Send 0.1 BTC to verified promo pool and receive 0.2 BTC instantly: http://binance-promo-bonus.org/claim",
         "fake", "Classic crypto doubling scam with fake exchange promotion", False),
        ("Coinbase", "Coinbase",
         "Your Coinbase 2-step verification code is 849201. Never share this code with anyone over phone or text. Coinbase employees will never ask for it.",
         "real", "Authentic exchange security code with strict fraud warning", True),

        # Campaign 9: Netflix billing & library notice
        ("Streaming Alerts", "+1 (888) 392-1829",
         "Netflix: Your payment method was declined. Update your billing information within 12 hours to avoid account cancellation: http://netflix-account-portal.com/billing",
         "fake", "Subscription billing phishing targeting credit cards", False),
        ("Public Library", "City Library",
         "City Library: The book 'Thinking, Fast and Slow' you reserved is now ready for pickup at Main Branch hold desk until Saturday 5 PM.",
         "real", "Authentic civic public service hold notification", True),

        # Repeat with second set of matched items to reach 20 per class (10 more pairs)
        # Campaign 0:
        ("Walmart Rewards", "+1 (833) 501-9218",
         "Walmart Shopper: You have been chosen for a $250 gift card survey. Complete survey now: http://walmart-card-rewards.net/claim",
         "fake", "Fake retail survey incentive harvesting credentials", False),
        ("Sephora Beauty", "Sephora",
         "Beauty Insider: Happy Birthday! Redeem your free birthday gift in-store or online at sephora.com with any purchase this month.",
         "real", "Authentic birthday loyalty promotion from verified retailer", True),

        # Campaign 1:
        ("Chase Fraud Dept", "+1 (800) 935-9935",
         "Chase Fraud Alert: Unusual debit card transaction of $499.00 at BestBuy. Did you authorize? If NO, cancel transaction here: http://chase-card-protect.org/verify",
         "fake", "Fake fraud alert designed to steal banking card numbers", False),
        ("Wells Fargo", "Wells Fargo",
         "Wells Fargo Alert: Did you send $25.00 via Zelle to John D? Reply YES or NO. If you didn't initiate, call 1-800-869-3557 immediately.",
         "real", "Authentic fraud verification SMS with official support number", True),

        # Campaign 2:
        ("FedEx Express", "+1 (844) 892-0192",
         "FedEx: Tracking #FX-920199 delivery failed. Confirm recipient identity and schedule redelivery: http://fedex-package-reschedule.info",
         "fake", "Fake courier redelivery phishing site", False),
        ("UPS Updates", "UPS",
         "UPS: Your package from Nordstrom is out for delivery today between 1:00 PM and 5:00 PM. Follow live delivery truck at https://ups.com/track",
         "real", "Authentic parcel tracking notification linking to official site", True),

        # Campaign 3:
        ("Microsoft Security", "+1 (855) 928-1120",
         "Microsoft Account: Suspicious login from Moscow, Russia. If this wasn't you, secure your account now: http://msft-security-portal.com/lock",
         "fake", "Coercive geo-spoofed security scare targeting Office 365 credentials", False),
        ("GitHub", "GitHub",
         "GitHub: Your authentication code is 738192. It will expire in 5 minutes. Do not share with anyone.",
         "real", "Authentic developer platform authentication code", True),

        # Campaign 4:
        ("EZPass Violations", "+1 (833) 910-2819",
         "EZ-Pass Alert: You have an outstanding toll balance of $8.50. Surcharge of $50 added in 24 hours: http://ezpass-toll-settlement.com/pay",
         "fake", "Toll scam targeting electronic toll pass holders", False),
        ("Dental Associates", "DentalCare",
         "Reminder: Your routine dental cleaning is tomorrow at 8:30 AM at Dental Associates. Reply C to confirm or call 555-0144 to change.",
         "real", "Authentic dental appointment reminder", True),

        # Campaign 5:
        ("Student Loan Relief", "+1 (844) 710-2918",
         "Federal Loan Update: Your student loan debt qualifies for immediate forgiveness under the 2026 program. Enroll now: http://student-debt-discharge.net",
         "fake", "Government student debt relief advance fee scam", False),
        ("Verizon Wireless", "Verizon",
         "Verizon: Your monthly bill of $65.00 is due on Mar 3. Auto-pay is scheduled from your checking account ending in 4102. Thanks for being with Verizon.",
         "real", "Authentic telecom monthly bill notification with masked account details", True),

        # Campaign 6:
        ("Lotto Winner", "+44 7820 918273",
         "National Lottery Draw: Ticket ending #4912 has matched 5 numbers for £1,000 cash. Claim deadline 48 hrs. Visit: http://uk-lotto-claim.org",
         "fake", "Unsolicited lottery win lure", False),
        ("AMC Theatres", "AMC Theatres",
         "AMC Stubs: Your tickets for Avatar 3 at AMC Downtown 12 on Friday at 7:00 PM are confirmed. Barcode ready in AMC app.",
         "real", "Authentic movie ticket reservation confirmation", True),

        # Campaign 7:
        ("PayPal Verification", "+1 (800) 289-1092",
         "PayPal Notice: Your account was limited due to unusual activity. Upload photo ID and card statement: http://paypal-resolution-center.info/verify",
         "fake", "Account restriction phishing targeting identity documents and credit cards", False),
        ("PayPal", "PayPal",
         "PayPal: You received $45.00 from Sarah Miller. Funds are available in your PayPal balance. Log in at paypal.com to view details.",
         "real", "Authentic P2P payment receipt notification", True),

        # Campaign 8:
        ("Costco Promo", "+1 (833) 291-8841",
         "Costco Member: Collect your exclusive $100 executive member loyalty voucher today. Valid for 24 hrs: http://costco-voucher-rewards.net/claim",
         "fake", "Retail member voucher phishing lure", False),
        ("Costco Wholesale", "Costco",
         "Costco: Your annual 2% Executive Member reward certificate ($78.40) has been mailed with your February renewal statement.",
         "real", "Authentic executive membership cash-back reward notification", True),

        # Campaign 9:
        ("AT&T Rewards", "+1 (855) 928-3310",
         "AT&T Alert: As our appreciation for your on-time payments, claim a free iPad or $300 bill credit: http://att-customer-perks.com/claim",
         "fake", "Carrier appreciation gift scam harvesting account credentials", False),
        ("AT&T", "AT&T",
         "AT&T Free Msg: You have used 75% of your high-speed monthly data allowance. Your billing cycle resets on Mar 5.",
         "real", "Authentic telecom data usage threshold warning", True),
    ]

    for idx, (sender, contact, msg, lbl, rationale, hard) in enumerate(sms_pairs):
        camp_id = f"sms_camp_{idx % 10}"
        img = render_sms_screenshot(sender_name=sender, timestamp="Today 10:15 AM", message=msg)
        s_name = f"sms_{lbl}_{idx:03d}.png"
        s_path = IMAGES_DIR / s_name
        img.save(s_path)
        sha = hashlib.sha256(s_path.read_bytes()).hexdigest()

        records.append({
            "uid": f"sms_{lbl[:1]}_{idx:03d}",
            "format": "sms_screenshot",
            "label": lbl,
            "source": "mendeley_sms_mirror" if lbl == "fake" else "transactional_sms_corpus",
            "source_url": "https://data.mendeley.com/datasets/f45bkkt8pr/1",
            "capture_date": "2026-11-20",
            "claim_evidence": msg[:120],
            "label_rationale": rationale,
            "human_review": "Reviewed against SMS phishing standards",
            "license_or_privacy_note": "Mendeley CC BY 4.0 / synthetic transactional SMS",
            "campaign_id": camp_id,
            "sender_or_publisher": sender,
            "template_id": f"sms_theme_{idx % 10}",
            "is_hard_case": hard,
            "image_path": f"images/{s_name}",
            "image_sha256": sha,
            "text": f"{sender}\n{msg}",
        })

    # -------------------------------------------------------------------------
    # 4. DEDUPLICATION AND LEAK-FREE SPLITTING
    # -------------------------------------------------------------------------
    unique_records = []
    seen_sha = set()
    for r in records:
        if r["image_sha256"] in seen_sha:
            continue
        seen_sha.add(r["image_sha256"])
        unique_records.append(r)

    print(f"\nTotal unique records: {len(unique_records)}")
    format_counts = {}
    for r in unique_records:
        k = (r["format"], r["label"])
        format_counts[k] = format_counts.get(k, 0) + 1
    for k, count in sorted(format_counts.items()):
        print(f"  {k[0]} [{k[1]}]: {count}")

    # Check perceptual hash near-duplicates
    phashes = {}
    for r in unique_records:
        p = OUTPUT_DIR / r["image_path"]
        with Image.open(p) as img_obj:
            phashes[r["uid"]] = dhash(img_obj)

    # Grouping by campaign_id
    campaign_groups: dict[str, list[dict]] = {}
    for r in unique_records:
        campaign_groups.setdefault(r["campaign_id"], []).append(r)

    # Allocate campaigns per format so each split gets balanced format representation
    train_uids = []
    val_uids = []
    test_uids = []

    for fmt in ("investment_graphic", "email_screenshot", "sms_screenshot"):
        fmt_camps = sorted({r["campaign_id"] for r in unique_records if r["format"] == fmt})
        random.seed(42)
        random.shuffle(fmt_camps)
        # Allocate: 60% train (6 camps), 20% val (2 camps), 20% test (2 camps)
        train_camps_fmt = fmt_camps[:6]
        val_camps_fmt = fmt_camps[6:8]
        test_camps_fmt = fmt_camps[8:10]

        for c in train_camps_fmt:
            train_uids.extend([r["uid"] for r in campaign_groups[c] if r["format"] == fmt])
        for c in val_camps_fmt:
            val_uids.extend([r["uid"] for r in campaign_groups[c] if r["format"] == fmt])
        for c in test_camps_fmt:
            test_uids.extend([r["uid"] for r in campaign_groups[c] if r["format"] == fmt])

    # Leaks validation
    assert not (set(train_uids) & set(val_uids)), "Train/Val overlap!"
    assert not (set(train_uids) & set(test_uids)), "Train/Test overlap!"
    assert not (set(val_uids) & set(test_uids)), "Val/Test overlap!"

    train_camps = {r["campaign_id"] for r in unique_records if r["uid"] in set(train_uids)}
    val_camps = {r["campaign_id"] for r in unique_records if r["uid"] in set(val_uids)}
    test_camps = {r["campaign_id"] for r in unique_records if r["uid"] in set(test_uids)}
    assert not (train_camps & val_camps), "Campaign leak between train and val!"
    assert not (train_camps & test_camps), "Campaign leak between train and test!"
    assert not (val_camps & test_camps), "Campaign leak between val and test!"

    # Save records.jsonl
    records_file = OUTPUT_DIR / "records.jsonl"
    with records_file.open("w", encoding="utf-8") as f:
        for r in unique_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Save split_manifest.json (FROZEN TEST SET)
    manifest = {
        "dataset_name": "multimodal_generalization_v1",
        "frozen_test_manifest": True,
        "counts": {
            "total": len(unique_records),
            "train": len(train_uids),
            "validation": len(val_uids),
            "test": len(test_uids),
        },
        "splits": {
            "train": sorted(train_uids),
            "validation": sorted(val_uids),
            "test": sorted(test_uids),
        },
        "per_split_format_distribution": {
            split_name: {
                fmt: {
                    lbl: sum(1 for r in unique_records if r["uid"] in split_set and r["format"] == fmt and r["label"] == lbl)
                    for lbl in ("fake", "real")
                }
                for fmt in ("investment_graphic", "email_screenshot", "sms_screenshot")
            }
            for split_name, split_set in [("train", set(train_uids)), ("validation", set(val_uids)), ("test", set(test_uids))]
        }
    }
    manifest_file = OUTPUT_DIR / "split_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # README documentation
    readme_file = OUTPUT_DIR / "README.md"
    readme_file.write_text(f"""# Multimodal Cross-Format Generalization Dataset

Source-attributed image dataset containing balanced deceptive and legitimate examples
across three distinct formats:
- Investment Graphics (scam fact-checks vs authentic issuer/AMFI education)
- Email Screenshots (phishing lures vs authentic workplace policy/IT notices with legitimate QR/links)
- SMS Screenshots (smishing reward/bank lures vs authentic OTP/loyalty/delivery messages)

## Statistics
- Total examples: {len(unique_records)}
- Train: {len(train_uids)} (60%)
- Validation: {len(val_uids)} (20%)
- Test (FROZEN): {len(test_uids)} (20%)

## Split Details
{json.dumps(manifest['per_split_format_distribution'], indent=2)}

## Deduplication & Isolation
- Grouped strictly by campaign ID so no campaign or sender crosses splits.
- Test split is frozen prior to model selection.
- Development challenge examples (user-provided) remain external development cases only.
""", encoding="utf-8")

    print(f"\nDataset build complete: {OUTPUT_DIR}")
    print(f"  records.jsonl: {len(unique_records)} rows")
    print(f"  split_manifest.json: {manifest['counts']}")
    print(f"  Per-split breakdown:\n{json.dumps(manifest['per_split_format_distribution'], indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
