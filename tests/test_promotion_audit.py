"""Behavior and evidence boundaries for the independent framing audit."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fipd.analysis.promotion_audit import audit_promotion


def evidence(**changes):
    item = {"claim_text": "Revenue grew 40%", "field": "text", "status": "supported",
            "source_url": "https://company.example/filing/2026-q1.pdf",
            "source_title": "Quarterly filing", "published_at": "2026-04-01",
            "reviewed_by": "Lab reviewer", "rationale": "Compared reported revenue for Q1.",
            "context": "The reviewer also identified a decline in operating profit."}
    item.update(changes)
    return item


def test_true_historical_fact_does_not_clear_independent_return_promise():
    text = "Revenue grew 40%. This stock will double. Buy now!"
    result = audit_promotion(text, as_of="2026-05-01", evidence=[evidence()])
    assert result["claims"][0]["status"] == "supported"
    assert {item["label"] for item in result["findings"]} == {"return_promise", "urgency"}
    assert result["claims"][0]["evidence_origin"] == "reviewer_supplied_not_independently_verified"
    assert "not fetched or independently verified" in result["observations"][0]
    assert "risk_score" not in result and "label" not in result
    json.dumps(result, allow_nan=False)


def test_offsets_are_exact_original_field_spans_and_do_not_leak_across_fields():
    text = "₹ First line. GUARANTEED 20% RETURNS!"
    caption = "Invest now in this stock. SEBI-registered."
    result = audit_promotion(text, caption)
    assert {row["field"] for row in result["findings"]} == {"text", "caption"}
    for row in result["findings"]:
        original = text if row["field"] == "text" else caption
        assert original[row["start"]:row["end"]] == row["quote"]
        assert row["reason"] and row["rule_id"]


@pytest.mark.parametrize("text", [
    "There are no guaranteed returns in stock investing.",
    "We cannot guarantee returns in stock trading.",
    "Returns are not guaranteed; capital is at risk.",
    "Guaranteed returns are not possible in the stock market.",
    "Not all returns are guaranteed in stock investing.",
    "Are returns guaranteed? No, stock investments can lose money.",
    "Is this a stock with guaranteed returns? Read the risk disclosure.",
    "Avoid promoters who promise guaranteed returns. Do not trust claims that a stock will double. "
    "Investments are subject to market risk.",
    'Scam warning: "Guaranteed returns! Buy now!"',
    "This stock could double if profits improve.",
    "This stock will rise if earnings improve.",
    "This investment may double your money, but losses are possible.",
    "If earnings improve, this stock will double.",
    "This stock will double, assuming earnings improve.",
    "This investment could potentially double your money.",
    "Guaranteed stock returns are a misleading promise.",
    "The stock rose 20% last year. Future performance is uncertain.",
])
def test_negation_education_conditionals_and_history_do_not_become_promises(text):
    assert not any(row["label"] == "return_promise"
                   for row in audit_promotion(text)["findings"])


def test_caution_in_one_sentence_does_not_clear_promise_in_next_sentence():
    result = audit_promotion("No investment is risk free. Guaranteed returns of 20%. Buy now.")
    assert any(row["label"] == "return_promise" for row in result["findings"])


@pytest.mark.parametrize("text", [
    "Buy this stock now for guaranteed double returns.",
    "This stock offers assured triple returns.",
    "Guaranteed 20% monthly returns on your investment.",
    "Guaranteed returns are not negotiable in this investment offer.",
    "Ignore the warning, guaranteed returns on this stock.",
    "This offer is not misleading: guaranteed returns on this stock.",
    "Investments carry a warning, but this stock will double.",
    "Guaranteed stock returns. This is not financial advice.",
    "This stock will double, read our risk warning before investing.",
    "This stock will double, although you could lose money.",
    "This stock will double, subject to market risk.",
    "If you buy, this stock will double.",
    "This stock will double if you invest now.",
    "Guaranteed returns, read the risk warning before investing.",
])
def test_common_return_guarantees_and_disclaimers_do_not_escape_detection(text):
    assert any(row["label"] == "return_promise" for row in audit_promotion(text)["findings"])


def test_advice_not_to_ignore_warning_stays_educational():
    assert not audit_promotion("Do not ignore the warning about guaranteed returns on stocks.")["findings"]


def test_ordinary_bullish_opinion_remains_without_fraud_or_future_price_verdict():
    result = audit_promotion("I believe this stock could perform well over five years. "
                             "Investments are subject to market risk.")
    assert result["findings"] == []
    assert any("not a finding" in row for row in result["observations"])
    assert any("no fraud verdict" in row for row in result["limitations"])


def test_missing_disclosures_are_only_scoped_observations_not_accusations():
    result = audit_promotion("This stock could grow.")
    assert result["findings"] == []
    assert "Unchecked does not mean false or supported" in result["observations"][0]
    assert any("may exist elsewhere" in row for row in result["observations"])
    assert any("does not establish that payment" in row for row in result["observations"])
    assert not any("selective_context" == row["label"] for row in result["findings"])


def test_sponsorship_wording_does_not_prove_payment_and_identity_is_not_verified():
    result = audit_promotion("#ad. SEBI-registered stock research. Thousands of investors have joined. "
                             "Capital at risk.")
    assert {row["label"] for row in result["findings"]} == {"authority_cue", "social_proof"}
    assert any("payment and conflicts have not been independently verified" in row
               for row in result["observations"])


def test_evidence_after_assessment_date_is_rejected_to_avoid_hindsight():
    with pytest.raises(ValueError, match="after as_of"):
        audit_promotion("Revenue grew 40%", as_of="2026-03-31", evidence=[evidence()])
    with pytest.raises(ValueError, match="Provide as_of"):
        audit_promotion("Revenue grew 40%", evidence=[evidence()])


def test_nonpresent_claim_or_forged_span_is_rejected():
    with pytest.raises(ValueError, match="exact quote"):
        audit_promotion("Revenue grew 4%", as_of="2026-05-01", evidence=[evidence()])
    with pytest.raises(ValueError, match="exact claim quote"):
        audit_promotion("Revenue grew 40%", as_of="2026-05-01",
                        evidence=[evidence(start=1, end=16)])
    with pytest.raises(ValueError, match="occurs more than once"):
        audit_promotion("Revenue grew 40%. Revenue grew 40%", as_of="2026-05-01",
                        evidence=[evidence()])
    result = audit_promotion("Revenue grew 40%. Revenue grew 40%", as_of="2026-05-01",
                             evidence=[evidence(start=18, end=34)])
    assert result["claims"][0]["start"] == 18


@pytest.mark.parametrize("changes,error", [
    ({"status": "true"}, "status"),
    ({"field": "article_title"}, "field"),
    ({"reviewed_by": ""}, "reviewed_by"),
    ({"published_at": "2026-02-30"}, "valid calendar date"),
    ({"source_url": "file:///C:/secrets"}, "public HTTP"),
    ({"source_url": "https://localhost/filing"}, "public HTTP"),
    ({"source_url": "https://127.0.0.1/filing"}, "private or loopback"),
    ({"source_url": "https://user:password@company.example/filing"}, "credentials"),
    ({"rationale": float("nan")}, "must be a string"),
    ({"future_price": 100}, "unsupported fields"),
])
def test_malformed_or_unattributed_reviews_fail_explicitly(changes, error):
    with pytest.raises(ValueError, match=error):
        audit_promotion("Revenue grew 40%", as_of="2026-05-01", evidence=[evidence(**changes)])


@pytest.mark.parametrize("kwargs,error", [
    ({"text": ""}, "Provide promotion text"),
    ({"text": float("nan")}, "text must be a string"),
    ({"text": "x" * 50_001}, "at most"),
    ({"text": "stock", "caption": []}, "caption must be a string"),
    ({"text": "stock", "as_of": "2026/05/01"}, "ISO date"),
    ({"text": "stock", "evidence": {}}, "evidence must be a list"),
    ({"text": "stock", "evidence": [{}] * 101}, "at most 100"),
])
def test_input_validation_and_limits(kwargs, error):
    with pytest.raises(ValueError, match=error):
        audit_promotion(**kwargs)


def test_caption_only_and_nonfinancial_advertising_do_not_get_stock_accusations():
    result = audit_promotion("", "This stock will double.")
    assert result["findings"][0]["field"] == "caption"
    assert audit_promotion("Buy now. Only 3 seats left for this concert.")["findings"] == []


@pytest.mark.parametrize("disclosure", ["You could lose money.", "Possible losses remain."])
def test_plain_loss_disclosure_is_reported_without_cancelling_promise(disclosure):
    result = audit_promotion("This stock will double. " + disclosure)
    assert any(row["label"] == "return_promise" for row in result["findings"])
    assert any("Risk-related wording was detected" in row for row in result["observations"])
