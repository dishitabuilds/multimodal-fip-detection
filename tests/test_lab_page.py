"""Exercise the actual lab page with real saved sklearn models."""
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
st = pytest.importorskip("streamlit")
pytest.importorskip("sklearn")
from streamlit.testing.v1 import AppTest
from tests.test_lab_models import lab_data  # noqa: F401
from fipd.training.lab import run_lab_experiment


@pytest.fixture
def lab_page(lab_data, monkeypatch):
    root, records, manifest, config = lab_data
    run, _ = run_lab_experiment(records, manifest, root, root / "out", config,
                                 models=["linear_svm"])
    original = Path.glob

    def glob(path, pattern, *args, **kwargs):
        if path == ROOT / "data/processed/lab/research":
            return original(run.parent, pattern, *args, **kwargs)
        if path == ROOT / "data/processed/lab/internal_smoke_only":
            return iter([])
        return original(path, pattern, *args, **kwargs)

    monkeypatch.setattr(Path, "glob", glob)
    st.cache_resource.clear()
    app = AppTest.from_file(str(ROOT / "pages/1_ML_Lab.py"), default_timeout=20).run()
    yield app
    st.cache_resource.clear()


def test_lab_page_classifies_and_clears_changed_input(lab_page):
    app = lab_page
    assert not app.exception
    assert len(app.selectbox) == 1 and len(app.dataframe) == 1
    assert app.button[0].disabled  # OCR needs an image
    app.text_area[0].set_value("guaranteed double profit telegram").run()
    app.button[1].click().run()
    assert not app.exception
    assert app.session_state["lab_prediction"]["label"] == "fake"
    assert any("not a probability" in item.value for item in app.caption)
    app.text_area[0].set_value("mutual fund market risk").run()
    assert "lab_prediction" not in app.session_state


def test_lab_page_handles_missing_text_and_empty_pilot_mode(lab_page):
    app = lab_page
    app.button[1].click().run()
    assert not app.exception
    assert any("provide OCR text" in item.value for item in app.error)
    app.checkbox[0].check().run()
    assert not app.exception
    assert len(app.selectbox) == 0
    assert any("No trained models" in item.value for item in app.info)
