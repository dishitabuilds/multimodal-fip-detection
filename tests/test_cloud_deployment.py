"""Cloud startup checks without model downloads or network access."""
from pathlib import Path

import pytest

from fipd.frontend import cloud


def prepare_files(root, relative, names):
    directory = root / relative
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"prepared")


def test_prepared_models_never_start_a_download(tmp_path, monkeypatch):
    prepare_files(tmp_path, "data/models/easyocr", cloud.OCR_FILES)
    prepare_files(tmp_path, "data/models/faster-whisper-base", cloud.SPEECH_FILES)
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **kw: pytest.fail("download"))
    cloud.prepare_cloud_assets(tmp_path, speech=True)


def test_image_startup_prepares_ocr_without_downloading_speech(tmp_path, monkeypatch):
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert kwargs["cwd"] == tmp_path.resolve()
        assert kwargs["check"] and kwargs["timeout"] == 900
        assert command[-1] == str(tmp_path / "data/models/easyocr")
        prepare_files(tmp_path, "data/models/easyocr", cloud.OCR_FILES)

    monkeypatch.setattr(cloud.subprocess, "run", run)
    cloud.prepare_cloud_assets(tmp_path)
    cloud.prepare_cloud_assets(tmp_path)
    assert len(commands) == 1
    assert not (tmp_path / "data/models/faster-whisper-base").exists()


def test_missing_and_empty_assets_are_retried_and_fail_clearly(tmp_path, monkeypatch):
    prepare_files(tmp_path, "data/models/easyocr", cloud.OCR_FILES)
    (tmp_path / "data/models/easyocr" / cloud.OCR_FILES[0]).write_bytes(b"")
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **kw: None)
    with pytest.raises(RuntimeError, match="required files"):
        cloud.prepare_cloud_assets(tmp_path)


def test_media_startup_uses_the_existing_speech_preparation_cli(tmp_path, monkeypatch):
    prepare_files(tmp_path, "data/models/easyocr", cloud.OCR_FILES)
    commands = []

    def run(command, **kwargs):
        commands.append(command)
        assert command[1:] == [str(tmp_path / "scripts/14_analyze_media.py"), "--prepare-model"]
        prepare_files(tmp_path, "data/models/faster-whisper-base", cloud.SPEECH_FILES)

    monkeypatch.setattr(cloud.subprocess, "run", run)
    cloud.prepare_cloud_assets(tmp_path, speech=True)
    cloud.prepare_cloud_assets(tmp_path, speech=True)
    assert len(commands) == 1


def test_failed_download_remains_retryable(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise cloud.subprocess.TimeoutExpired("model setup", 900)

    monkeypatch.setattr(cloud.subprocess, "run", fail)
    with pytest.raises(RuntimeError, match="model preparation failed"):
        cloud.prepare_cloud_assets(tmp_path)
    monkeypatch.setattr(cloud.subprocess, "run", lambda *a, **kw:
                        prepare_files(tmp_path, "data/models/easyocr", cloud.OCR_FILES))
    cloud.prepare_cloud_assets(tmp_path)


def test_cloud_entrypoint_preserves_the_existing_image_page(monkeypatch):
    st = pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    st.cache_resource.clear()
    monkeypatch.setattr(cloud, "prepare_cloud_assets", lambda *a, **kw: None)
    root = Path(__file__).resolve().parents[1]
    app = AppTest.from_file(str(root / "deployment/streamlit_app.py")).run(timeout=30)
    assert not app.exception
    assert app.selectbox(key="analysis_mode").value == "Extract text"
    st.cache_resource.clear()
