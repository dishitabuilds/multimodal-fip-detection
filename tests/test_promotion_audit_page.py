"""The framing page works without an authenticity model or external evidence fetch."""
from datetime import date
import json
import io
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest


@pytest.fixture
def audit_page(monkeypatch):
    original = Path.glob

    def glob(path, pattern, *args, **kwargs):
        if path == ROOT / "data/processed/framing/research":
            return iter([])
        return original(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    st.cache_resource.clear()
    page = AppTest.from_file(str(ROOT / "pages/2_Promotion_Audit.py"),
                             default_timeout=20).run()
    assert not page.exception
    yield page
    st.cache_resource.clear()


def run_audit(page, text):
    page.text_area(key="audit_text").set_value(text).run()
    page.button(key="audit_run").click().run()
    assert not page.exception
    return page.session_state["promotion_audit"]


def test_standalone_page_flags_promises_but_not_a_false_fact(audit_page):
    page = audit_page
    assert page.button(key="audit_read_image").disabled
    assert any("No reviewed-data framing model" in item.value for item in page.caption)
    text = "Revenue grew 40% last year. Buy this stock now for guaranteed double returns."
    result = run_audit(page, text)
    assert any(item["label"] == "return_promise" for item in result["findings"])
    assert result["claims"] == []
    for item in result["findings"]:
        assert text[item["start"]:item["end"]] == item["quote"]
    assert any("Facts have not been checked" in item.value for item in page.info)
    assert any("Future stock performance" in item.value for item in page.markdown)


def test_hypothetical_warning_is_not_treated_as_a_promise(audit_page):
    page = audit_page
    page.selectbox(key="audit_example").select("Hypothetical investor warning").run()
    page.button(key="audit_load_example").click().run()
    page.button(key="audit_run").click().run()
    assert not page.exception
    result = page.session_state["promotion_audit"]
    assert not any(item["label"] == "return_promise" for item in result["findings"])


def test_claim_evidence_is_separate_and_future_evidence_rejected(audit_page):
    page = audit_page
    text = "Revenue grew 40% last year."
    page.text_area(key="audit_text").set_value(text).run()
    page.date_input(key="audit_as_of").set_value(date(2026, 5, 1)).run()
    evidence = [{"claim_text": text, "field": "text", "status": "supported",
                 "source_url": "https://example.com/report", "source_title": "Annual report",
                 "published_at": "2026-04-01", "reviewed_by": "Test reviewer",
                 "rationale": "Fixture review of the same company and reporting period."}]
    page.text_area(key="audit_evidence").set_value(json.dumps(evidence)).run()
    page.button(key="audit_run").click().run()
    assert not page.exception
    assert page.session_state["promotion_audit"]["claims"][0]["status"] == "supported"
    evidence[0]["published_at"] = "2026-06-01"
    page.text_area(key="audit_evidence").set_value(json.dumps(evidence)).run()
    assert "promotion_audit" not in page.session_state
    page.button(key="audit_run").click().run()
    assert not page.exception
    assert page.error
    assert "promotion_audit" not in page.session_state


@pytest.mark.parametrize("key,value", [
    ("audit_text", "Markets are uncertain."),
    ("audit_caption", "A different caption"),
    ("audit_evidence", "not valid JSON"),
])
def test_changed_input_removes_the_previous_result(audit_page, key, value):
    page = audit_page
    run_audit(page, "Guaranteed double returns.")
    page.text_area(key=key).set_value(value).run()
    assert not page.exception
    assert "promotion_audit" not in page.session_state


def test_date_change_and_invalid_json_cannot_leave_stale_claim_evidence(audit_page):
    page = audit_page
    run_audit(page, "Guaranteed double returns.")
    page.date_input(key="audit_as_of").set_value(date(2020, 1, 1)).run()
    assert "promotion_audit" not in page.session_state
    page.text_area(key="audit_evidence").set_value("{}").run()
    page.button(key="audit_run").click().run()
    assert not page.exception
    assert any("must be a list" in item.value for item in page.error)
    assert "promotion_audit" not in page.session_state


def test_empty_inputs_fail_with_an_explanation(audit_page):
    page = audit_page
    page.button(key="audit_run").click().run()
    assert not page.exception
    assert page.error
    assert "promotion_audit" not in page.session_state


@pytest.mark.parametrize("replacement_is_valid", [True, False])
def test_new_image_and_removal_clear_transcript_context_and_output(
        audit_page, monkeypatch, replacement_is_valid):
    from PIL import Image

    page = audit_page
    run_audit(page, "Guaranteed double returns.")
    page.text_area(key="audit_caption").set_value("Caption for the old creative").run()
    review = [{"claim_text": "Guaranteed double returns.", "field": "text",
               "status": "insufficient_evidence", "source_url": "https://example.com/report",
               "source_title": "Fixture report", "published_at": "2000-01-01",
               "reviewed_by": "Test reviewer", "rationale": "Fixture evidence."}]
    page.text_area(key="audit_evidence").set_value(json.dumps(review)).run()
    page.button(key="audit_run").click().run()
    assert "promotion_audit" in page.session_state
    image_bytes = io.BytesIO()
    Image.new("RGB", (32, 32), "white").save(image_bytes, format="PNG")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: image_bytes)
    page.run()
    assert not page.exception
    assert page.text_area(key="audit_text").value == ""
    assert page.text_area(key="audit_caption").value == ""
    assert page.text_area(key="audit_evidence").value == "[]"
    assert "promotion_audit" not in page.session_state
    assert not page.button(key="audit_read_image").disabled
    run_audit(page, "Guaranteed double returns.")
    page.text_area(key="audit_caption").set_value("Caption for this image").run()
    page.text_area(key="audit_evidence").set_value(json.dumps(review)).run()
    page.button(key="audit_run").click().run()
    assert "promotion_audit" in page.session_state
    replacement = io.BytesIO()
    if replacement_is_valid:
        Image.new("RGB", (32, 32), "blue").save(replacement, format="PNG")
    else:
        replacement.write(b"not a valid image")
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: replacement)
    page.run()
    assert not page.exception
    assert page.text_area(key="audit_text").value == ""
    assert page.text_area(key="audit_caption").value == ""
    assert page.text_area(key="audit_evidence").value == "[]"
    assert "promotion_audit" not in page.session_state
    run_audit(page, "Guaranteed double returns.")
    page.text_area(key="audit_caption").set_value("Another caption").run()
    page.text_area(key="audit_evidence").set_value(json.dumps(review)).run()
    page.button(key="audit_run").click().run()
    assert "promotion_audit" in page.session_state
    monkeypatch.setattr(st, "file_uploader", lambda *args, **kwargs: None)
    page.run()
    assert not page.exception
    assert page.text_area(key="audit_text").value == ""
    assert page.text_area(key="audit_caption").value == ""
    assert page.text_area(key="audit_evidence").value == "[]"
    assert "promotion_audit" not in page.session_state
    assert page.button(key="audit_read_image").disabled


def test_learned_labels_are_optional_and_separate(audit_page, monkeypatch, tmp_path):
    from fipd.training import framing

    page = audit_page
    model_path = tmp_path / "reviewed-run/models/framing_linear_svm.joblib"
    model_path.parent.mkdir(parents=True)
    model_path.write_bytes(b"UI-only placeholder; mocked loader does not deserialize this")
    original = Path.glob

    def glob(path, pattern, *args, **kwargs):
        if path == ROOT / "data/processed/framing/research":
            return iter([model_path])
        return original(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    monkeypatch.setattr(framing, "load_framing_artifact", lambda path: {"fixture": True})
    monkeypatch.setattr(framing, "predict_framing", lambda *args: {
        "labels": ["return_promise"], "margins": {"return_promise": 0.25},
        "threshold": 0.0, "score_kind": "decision_margin", "notice": "Fixture model notice."})
    page.run()
    page.checkbox[0].check().run()
    result = run_audit(page, "Guaranteed returns.")
    assert result["learned_framing"]["labels"] == ["return_promise"]
    assert any("Margins are not probabilities" in item.value for item in page.caption)
    assert not result["claims"]
    page.checkbox[0].uncheck().run()
    assert "promotion_audit" not in page.session_state
