"""Offline AppTest coverage for the input workspace and honest analysis modes.

The actual Streamlit script handles every rerun. A shared service stub supplies
results without OCR/model downloads. AppTest has no upload setter, so that
widget alone is replaced with fresh in-memory PNG files. Backend inference and
temporary-file cleanup have their own service tests.
"""

from __future__ import annotations

import hashlib
import io
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "app.py"
sys.path.insert(0, str(ROOT / "src"))

st = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402
from fipd.frontend import analysis as analysis_service  # noqa: E402

OCR_TEXT = "SEBI registered investment. Guaranteed 40% monthly returns."


def png_bytes(color):
    output = io.BytesIO()
    Image.new("RGB", (48, 32), color).save(output, format="PNG")
    return output.getvalue()


def image_digest(image):
    return hashlib.sha256(image.convert("RGB").tobytes()).hexdigest()


def previews(app):
    return app.get("image") or app.get("imgs")


def preview_bytes(app):
    return [item.proto.SerializeToString() for item in previews(app)
            if any(image.caption == "Selected promotion" for image in item.proto.imgs)]


def downloads(app):
    return [item for item in app.get("download_button")
            if item.proto.label == "Download analysis"]


def result(app):
    return app.session_state["analysis_result"] if "analysis_result" in app.session_state else None


def assert_no_prediction(app):
    rendered = "\n".join(item.value for item in app.markdown).lower()
    assert "likely fake" not in rendered
    assert "likely real" not in rendered
    assert "model prediction" not in rendered
    assert 'class="score-strip"' not in rendered
    assert "assigned class:" not in rendered


@pytest.fixture(autouse=True)
def clear_resource_cache():
    st.cache_resource.clear()
    yield
    st.cache_resource.clear()


@pytest.fixture
def workspace(monkeypatch):
    presence = {"research": False, "pilot": False}
    stamps = {"research": 1, "pilot": 1}
    paths = {
        (ROOT / analysis_service.RESEARCH_CHECKPOINT_RELATIVE).absolute(): "research",
        (ROOT / analysis_service.PILOT_CHECKPOINT_RELATIVE).absolute(): "pilot",
    }
    for method in ("exists", "is_file"):
        original = getattr(Path, method)

        def check(path, _original=original):
            mode = paths.get(path.absolute())
            return presence[mode] if mode else _original(path)

        monkeypatch.setattr(Path, method, check)
    original_stat = Path.stat

    def stat(path, *args, **kwargs):
        mode = paths.get(path.absolute())
        if mode:
            if not presence[mode]:
                raise FileNotFoundError(path)
            return SimpleNamespace(st_mtime_ns=stamps[mode], st_size=1234)
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", stat)
    uploads, upload_keys, calls = {}, [], []
    failures = {"code": None}

    def uploader(*args, **kwargs):
        key = kwargs["key"]
        upload_keys.append(key)
        if key not in uploads:
            return None
        uploaded = io.BytesIO(uploads[key])
        uploaded.name = "offline-upload.png"
        uploaded.type = "image/png"
        uploaded.size = len(uploads[key])
        return uploaded

    monkeypatch.setattr(st, "file_uploader", uploader)

    def analyze(image, caption="", mode="ocr", root=ROOT, **kwargs):
        calls.append({"mode": mode, "caption": caption, "image": image_digest(image)})
        prediction = mode != "ocr" and not failures["code"]
        payload = {
            "status": "error" if failures["code"] else "complete", "mode": mode,
            "model_status": ("unavailable" if failures["code"] else "not_loaded" if mode == "ocr"
                             else "unvalidated_internal_pilot" if mode == "pilot"
                             else "compatible_checkpoint"),
            "purpose": {"ocr": "text_extraction_only", "pilot": "internal_smoke_only",
                        "research": "research_inference"}[mode],
            "raw_ocr": ("" if failures["code"] == "ocr_failure" else
                        "Partial OCR from the current image" if failures["code"] else OCR_TEXT),
            "ocr": (None if failures["code"] == "ocr_failure" else
                    {"backend": "easyocr", "lines": [OCR_TEXT], "mean_confidence": 0.9,
                     "elapsed_ms": 10.0, "script_profile": {"latin": 1.0}}),
            "indicators": [], "class_scores": {"real": 0.22, "fake": 0.78} if prediction else None,
            "threshold": 0.5 if prediction else None, "label": "fake" if prediction else None,
            "durations_ms": {"ocr_load": 1.0, "ocr": 10.0, "classifier_load": 2.0,
                             "inference": 3.0, "total": 16.0},
            "input_fingerprint": analysis_service.input_fingerprint(image, caption, mode),
            "warnings": ["Unvalidated internal pilot; no verified accuracy."] if mode == "pilot" else [],
            "error": ({"code": failures["code"], "message": "Offline analysis failed cleanly",
                       "details": "Test failure: no model downloads were attempted"}
                      if failures["code"] else None),
            "input": {"width": image.width, "height": image.height, "caption_chars": len(caption)},
        }
        return deepcopy(payload)

    analyze_mock = Mock(side_effect=analyze)
    ocr_loader = Mock(side_effect=AssertionError("UI browsing must not initialize OCR"))
    model_loader = Mock(side_effect=AssertionError("UI browsing must not initialize a model"))
    monkeypatch.setattr(analysis_service, "analyze_image", analyze_mock)
    monkeypatch.setattr(analysis_service, "load_ocr", ocr_loader)
    monkeypatch.setattr(analysis_service, "load_classifier", model_loader)
    return SimpleNamespace(presence=presence, stamps=stamps, uploads=uploads,
                           upload_keys=upload_keys, calls=calls, failures=failures,
                           analyze=analyze_mock, ocr_loader=ocr_loader, model_loader=model_loader)


def start():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception
    return app


def select_sample(app, key="sample_public_figure"):
    if app.radio(key="input_source").value != "Sample gallery":
        if any(button.key == "open_samples" for button in app.button):
            app.button(key="open_samples").click().run()
        else:
            app.radio(key="input_source").set_value("Sample gallery").run()
    app.button(key=key).click().run()
    assert not app.exception
    return app


def analyze(app):
    assert not app.button(key="analyze").disabled
    app.button(key="analyze").click().run()
    assert not app.exception


def test_preview_and_navigation_work_without_models(workspace):
    app = start()
    assert app.radio(key="input_source").value == "Upload"
    assert app.selectbox(key="analysis_mode").value == "Extract text"
    assert app.button(key="analyze").disabled
    assert not result(app)
    workspace.uploads["upload_0"] = png_bytes("red")
    app.run()
    assert not app.exception
    assert previews(app)
    assert not app.button(key="analyze").disabled
    workspace.analyze.assert_not_called()
    workspace.ocr_loader.assert_not_called()
    workspace.model_loader.assert_not_called()
    assert_no_prediction(app)


def test_samples_clear_caption_and_change_preview_from_other_cwd(workspace, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    app = select_sample(start())
    first_selection = app.session_state["selected_sample"]
    first_preview = preview_bytes(app)
    assert first_preview
    assert app.text_area(key="caption_input").value == ""
    analyze(app)
    first_image = workspace.calls[-1]["image"]
    app.text_area(key="caption_input").input("An editable user caption").run()
    assert app.text_area(key="caption_input").value == "An editable user caption"
    app.button(key="sample_fund_comparison").click().run()
    assert not app.exception
    assert app.session_state["selected_sample"] != first_selection
    assert app.text_area(key="caption_input").value == ""
    assert preview_bytes(app) != first_preview
    assert not result(app)
    analyze(app)
    assert workspace.calls[-1]["image"] != first_image


def test_explicit_source_controls_image_precedence(workspace):
    red = png_bytes("red")
    workspace.uploads["upload_0"] = red
    app = select_sample(start(), "sample_news_page")
    app.text_area(key="caption_input").input("The user's original caption").run()
    analyze(app)
    with Image.open(io.BytesIO(red)) as image:
        uploaded_digest = image_digest(image)
    assert workspace.calls[-1]["image"] != uploaded_digest
    assert workspace.calls[-1]["caption"] == "The user's original caption"
    app.radio(key="input_source").set_value("Upload").run()
    assert not result(app)
    analyze(app)
    assert workspace.calls[-1]["image"] == uploaded_digest


def test_ocr_only_works_without_checkpoint_and_has_no_prediction(workspace):
    app = select_sample(start(), "sample_goal_planning")
    analyze(app)
    assert workspace.calls[-1]["mode"] == "ocr"
    assert result(app)["class_scores"] is None
    assert result(app)["threshold"] is None
    assert result(app)["label"] is None
    assert any(OCR_TEXT in code.value for code in app.code)
    assert downloads(app)
    workspace.model_loader.assert_not_called()
    assert_no_prediction(app)


def test_results_persist_only_for_unchanged_inputs_and_checkpoint(workspace):
    workspace.uploads["upload_0"] = png_bytes("red")
    workspace.presence["pilot"] = True
    app = start()
    analyze(app)
    initial = deepcopy(result(app))
    app.run()
    assert result(app) == initial
    app.text_area(key="caption_input").input("A changed caption").run()
    assert not result(app)
    assert not downloads(app)
    analyze(app)
    workspace.uploads["upload_0"] = png_bytes("blue")
    app.run()
    assert not result(app)
    analyze(app)
    app.selectbox(key="analysis_mode").select("Pilot demo").run()
    assert not result(app)
    analyze(app)
    assert result(app)["mode"] == "pilot"
    workspace.stamps["pilot"] += 1
    app.run()
    assert not result(app)
    assert not downloads(app)


def test_pilot_mode_warns_before_and_after_analysis(workspace):
    workspace.presence["pilot"] = True
    app = select_sample(start())
    app.selectbox(key="analysis_mode").select("Pilot demo").run()
    assert any("provisional" in warning.value.lower() for warning in app.warning)
    assert any("pilot" in item.value.lower() for item in app.markdown)
    analyze(app)
    assert workspace.calls[-1]["mode"] == "pilot"
    assert result(app)["model_status"] == "unvalidated_internal_pilot"
    assert result(app)["purpose"] == "internal_smoke_only"
    assert any("pilot" in warning.value.lower() for warning in app.warning)
    assert downloads(app)


def test_missing_research_checkpoint_disables_only_research(workspace):
    app = select_sample(start())
    app.selectbox(key="analysis_mode").select("Research model").run()
    assert not app.exception
    assert app.button(key="analyze").disabled
    assert any("checkpoint" in item.value.lower() for item in [*app.info, *app.warning])
    workspace.analyze.assert_not_called()
    app.selectbox(key="analysis_mode").select("Extract text").run()
    assert not app.button(key="analyze").disabled
    assert_no_prediction(app)


@pytest.mark.parametrize("error_code", ["ocr_failure", "inference_failure"])
def test_analysis_errors_remove_previous_success(workspace, error_code):
    app = select_sample(start())
    analyze(app)
    assert downloads(app)
    workspace.failures["code"] = error_code
    app.text_area(key="caption_input").input("Changed input before retry").run()
    analyze(app)
    assert any("Offline analysis failed cleanly" in error.value for error in app.error)
    assert result(app)["status"] == "error"
    assert bool(downloads(app)) == (error_code == "inference_failure")
    assert not any(OCR_TEXT in code.value for code in app.code)
    assert_no_prediction(app)


def test_clear_workspace_resets_source_caption_upload_and_result(workspace):
    workspace.uploads["upload_0"] = png_bytes("red")
    app = select_sample(start())
    app.text_area(key="caption_input").input("A temporary caption").run()
    analyze(app)
    app.button(key="reset_input").click().run()
    assert not app.exception
    assert app.radio(key="input_source").value == "Upload"
    assert app.text_area(key="caption_input").value == ""
    assert not app.session_state["selected_sample"]
    assert not result(app)
    assert not downloads(app)
    assert app.button(key="analyze").disabled
    assert "upload_1" in workspace.upload_keys


def test_malformed_upload_clears_prior_result_and_never_calls_service(workspace):
    workspace.uploads["upload_0"] = png_bytes("red")
    app = start()
    analyze(app)
    assert downloads(app)
    workspace.analyze.reset_mock()
    workspace.uploads["upload_0"] = b"<html>This is not an image</html>"
    app.run()
    assert not app.exception
    assert app.error
    assert app.button(key="analyze").disabled
    assert not result(app)
    assert not downloads(app)
    workspace.analyze.assert_not_called()
    workspace.ocr_loader.assert_not_called()
    workspace.model_loader.assert_not_called()


def test_oversized_dimensions_are_rejected_before_analysis(workspace):
    output = io.BytesIO()
    # One-bit solid PNG stays small on disk while its actual geometry exceeds
    # the 12M-pixel guard. No giant RGB tensor or model is needed for this case.
    Image.new("1", (4000, 3001)).save(output, format="PNG")
    workspace.uploads["upload_0"] = output.getvalue()
    app = start()
    assert app.error
    assert app.button(key="analyze").disabled
    assert not result(app)
    assert not previews(app)
    workspace.analyze.assert_not_called()
    workspace.ocr_loader.assert_not_called()
    workspace.model_loader.assert_not_called()
