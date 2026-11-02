"""Offline image analysis regressions; no OCR/model downloads or training."""

from __future__ import annotations

import builtins
import copy
from dataclasses import asdict
import io
import json
from pathlib import Path
import sys

import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fipd.datasets.examples import INPUT_PIPELINE_VERSION  # noqa: E402
from fipd.enrichment.ocr import OCRResult  # noqa: E402
from fipd.frontend import analysis  # noqa: E402
from fipd.models.config import load_model_config  # noqa: E402


class FakeOCR:
    def __init__(self, text="Guaranteed returns of ₹22,000 monthly. Join WhatsApp.", error=""):
        self.text, self.error, self.paths = text, error, []

    def read(self, path):
        path = Path(path)
        assert path.is_file()
        with Image.open(path) as image:
            assert image.mode == "RGB"
        self.paths.append(path)
        return OCRResult(text=self.text, lines=[self.text] if self.text else [],
                         confidences=[0.9], backend="offline-test", elapsed_ms=2.0,
                         error=self.error)


def test_ocr_only_never_loads_or_imports_classifier(monkeypatch):
    ocr = FakeOCR()

    def forbidden(*args, **kwargs):
        pytest.fail("OCR-only must not initialize a classifier")

    original_import = builtins.__import__

    def block_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            pytest.fail("injected OCR-only analysis must not import torch")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", block_torch)
    monkeypatch.setattr(analysis, "load_classifier", forbidden)
    result = analysis.analyze_image(Image.new("RGB", (24, 18)), "SEBI mentioned", mode="ocr",
                                    ocr_loader=lambda: ocr, model_loader=forbidden)
    assert result["status"] == "complete"
    assert result["raw_ocr"] == ocr.text
    assert result["class_scores"] is result["label"] is result["threshold"] is None
    assert result["model_status"] == "not_loaded"
    assert result["purpose"] == "text_extraction_only"
    assert {cue["id"] for cue in result["indicators"]} >= {"certainty", "amounts", "authority"}
    assert all("does not establish" in cue["explanation"] for cue in result["indicators"])
    assert all(not path.exists() for path in ocr.paths)
    assert all(value >= 0 for value in result["durations_ms"].values())
    json.dumps(result, allow_nan=False)


def test_fingerprint_tracks_caption_mode_and_pixels():
    image = Image.new("RGB", (2, 3), "red")
    digest = analysis.input_fingerprint(image, "caption", "ocr")
    assert digest == analysis.input_fingerprint(image.copy(), "caption", "ocr")
    assert digest != analysis.input_fingerprint(image, "caption ", "ocr")
    assert digest != analysis.input_fingerprint(image, "caption", "pilot")
    changed = image.copy()
    changed.putpixel((0, 0), (0, 0, 0))
    assert digest != analysis.input_fingerprint(changed, "caption", "ocr")


def test_alpha_composited_on_white_without_mutating_original():
    image = Image.new("RGBA", (2, 1), (255, 0, 0, 0))
    image.putpixel((1, 0), (0, 0, 0, 128))
    before = image.tobytes()
    normalized = analysis.canonical_image(image)
    assert normalized.mode == "RGB"
    assert normalized.getpixel((0, 0)) == (255, 255, 255)
    assert normalized.getpixel((1, 0)) == (127, 127, 127)
    assert image.mode == "RGBA" and image.tobytes() == before
    assert analysis.input_fingerprint(Image.new("RGBA", (2, 1), (0, 0, 255, 0))) == \
        analysis.input_fingerprint(Image.new("RGB", (2, 1), "white"))


def test_exif_orientation_matches_analyzed_pixels():
    image = Image.new("RGB", (3, 2), "white")
    image.putpixel((0, 0), (255, 0, 0))
    exif = Image.Exif()
    exif[274] = 6
    memory = io.BytesIO()
    image.save(memory, format="JPEG", exif=exif)
    memory.seek(0)
    with Image.open(memory) as oriented:
        normalized = analysis.canonical_image(oriented)
        assert normalized.size == (2, 3)
        assert analysis.input_fingerprint(oriented) == analysis.input_fingerprint(normalized)


def test_mode_runtime_never_loads_config_or_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("runtime metadata must not initialize anything")

    monkeypatch.setattr(analysis, "load_model_config", forbidden)
    assert analysis.mode_runtime(tmp_path, "ocr")["checkpoint_path"] is None
    research = analysis.mode_runtime(tmp_path, "research")
    assert not research["checkpoint_available"]
    assert research["checkpoint_path"] == str(tmp_path / analysis.RESEARCH_CHECKPOINT_RELATIVE)
    path = tmp_path / analysis.PILOT_CHECKPOINT_RELATIVE
    path.parent.mkdir(parents=True)
    path.write_bytes(b"presence only")
    pilot = analysis.mode_runtime(tmp_path, "pilot")
    assert pilot["checkpoint_available"] and pilot["unvalidated"]
    assert pilot["purpose"] == "internal_smoke_only"


def test_load_ocr_uses_only_local_cpu_models(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from fipd.enrichment import ocr as ocr_module

    model_dir = tmp_path / "data/models/easyocr"
    model_dir.mkdir(parents=True)
    captured = {}

    class Engine:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.initializations = 0

        def initialize(self):
            self.initializations += 1

    monkeypatch.setattr(ocr_module, "EasyOCREngine", Engine)
    threads = []
    monkeypatch.setattr(torch, "set_num_threads", threads.append)
    engine = analysis.load_ocr(tmp_path)
    assert captured == {"languages": ["en", "hi"], "gpu": False,
                        "model_storage_directory": str(model_dir), "download_enabled": False}
    assert engine.initializations == 1 and threads == [4]


def test_missing_assets_fail_before_initialization(tmp_path, monkeypatch):
    with pytest.raises(analysis.AnalysisError) as error:
        analysis.load_ocr(tmp_path)
    assert error.value.code == "ocr_models_missing"

    def forbidden(*args, **kwargs):
        pytest.fail("a missing checkpoint must fail before config/encoder loading")

    monkeypatch.setattr(analysis, "load_model_config", forbidden)
    for mode in ("research", "pilot"):
        with pytest.raises(analysis.AnalysisError) as error:
            analysis.load_classifier(tmp_path, mode)
        assert error.value.code == "checkpoint_missing"


@pytest.fixture
def local_classifier(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    hub = pytest.importorskip("huggingface_hub")
    from fipd.models import arms

    captured = {"build": [], "tokenizer": [], "snapshots": []}

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = torch.nn.Linear(2, 2)

    def build(arm, architecture, **kwargs):
        captured["build"].append((arm, architecture, kwargs))
        assert Path(architecture.text_encoder).is_absolute()
        assert Path(architecture.image_encoder).is_absolute()
        assert kwargs == {"from_scratch": True}
        return Model()

    def tokenizer(path, **kwargs):
        captured["tokenizer"].append((path, kwargs))
        assert kwargs == {"local_files_only": True}
        return "offline-tokenizer"

    def snapshot(identifier, **kwargs):
        captured["snapshots"].append((identifier, kwargs))
        assert kwargs == {"local_files_only": True}
        return str(tmp_path / "snapshots" / ("text" if "distilbert" in identifier else "image"))

    monkeypatch.setattr(arms, "build_arm", build)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(hub, "snapshot_download", snapshot)

    def prepare(mode, saved_purpose=None):
        relative = analysis.PILOT_CONFIG_RELATIVE if mode == "pilot" else analysis.RESEARCH_CONFIG_RELATIVE
        cfg = load_model_config(ROOT / relative)
        path = tmp_path / relative
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text("# fixture", encoding="utf-8")
        monkeypatch.setattr(analysis, "load_model_config", lambda path: copy.deepcopy(cfg))
        directories = (tmp_path / cfg.student.text_encoder, tmp_path / cfg.student.image_encoder) \
            if mode == "pilot" else (tmp_path / "snapshots/text", tmp_path / "snapshots/image")
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "config.json").write_text("{}", encoding="utf-8")
        path = tmp_path / (analysis.PILOT_CHECKPOINT_RELATIVE if mode == "pilot"
                           else analysis.RESEARCH_CHECKPOINT_RELATIVE)
        path.parent.mkdir(parents=True, exist_ok=True)
        purpose = cfg.raw.get("purpose") if saved_purpose is None else saved_purpose
        checkpoint = {"state_dict": Model().state_dict(), "arm": "fused", "variant": "student",
                      "threshold": 0.37, "experiment_id": "offline-test",
                      "config": {"input_pipeline_version": INPUT_PIPELINE_VERSION,
                                 "architecture": asdict(cfg.student), "text": asdict(cfg.text),
                                 "measurement": asdict(cfg.measurement),
                                 "source_config": {"purpose": purpose}}}
        torch.save(checkpoint, path)
        return cfg, path

    return prepare, captured, tmp_path


@pytest.mark.parametrize("mode", ["research", "pilot"])
def test_contract_compared_before_paths_resolved_locally(local_classifier, mode):
    prepare, captured, root = local_classifier
    cfg, path = prepare(mode)
    components = analysis.load_classifier(root, mode)
    assert components.mode == mode and components.threshold == 0.37
    assert components.checkpoint_path == path
    assert components.config.student.text_encoder == cfg.student.text_encoder
    assert not components.model.training
    assert len(captured["build"]) == len(captured["tokenizer"]) == 1
    assert len(captured["snapshots"]) == (0 if mode == "pilot" else 2)
    assert tuple(components.transform(Image.new("RGB", (8, 9))).shape) == (3, 224, 224)


@pytest.mark.parametrize("mode,purpose", [("pilot", "research"), ("research", "internal_smoke_only")])
def test_pilot_and_research_provenance_cannot_be_swapped(local_classifier, mode, purpose):
    prepare, captured, root = local_classifier
    prepare(mode, saved_purpose=purpose)
    with pytest.raises(analysis.AnalysisError) as error:
        analysis.load_classifier(root, mode)
    assert error.value.code == "checkpoint_incompatible"
    assert not captured["build"] and not captured["snapshots"]


def test_incompatible_tokenizer_rejected_before_asset_resolution(local_classifier):
    torch = pytest.importorskip("torch")
    prepare, captured, root = local_classifier
    _, path = prepare("research")
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["config"]["architecture"]["text_encoder"] = "different-same-shape-tokenizer"
    torch.save(checkpoint, path)
    with pytest.raises(analysis.AnalysisError) as error:
        analysis.load_classifier(root, "research")
    assert error.value.code == "checkpoint_incompatible"
    assert not captured["snapshots"] and not captured["build"]


def components_for_test(mode, logits=None, threshold=0.8, fail=False):
    torch = pytest.importorskip("torch")
    cfg = load_model_config()
    cfg.text.transliterate = False
    cfg.text.fields = ["ocr_text", "claim_text"]
    captured = {}

    class Model(torch.nn.Module):
        def forward(self, **kwargs):
            captured["model_inputs"] = kwargs
            if fail:
                raise RuntimeError("PRIVATE INPUT MUST NOT BE EXPOSED")
            return torch.tensor(logits if logits is not None else [[0.0, 1.0]])

    def tokenizer(text, **kwargs):
        captured["text"] = text
        return {"input_ids": torch.ones(1, 8, dtype=torch.long),
                "attention_mask": torch.ones(1, 8, dtype=torch.long)}

    components = analysis.ClassifierComponents(
        model=Model().eval(), tokenizer=tokenizer,
        transform=lambda image: torch.zeros(3, 224, 224), config=cfg,
        threshold=threshold, mode=mode, purpose=analysis.MODE_PURPOSES[mode],
        checkpoint_path=Path("offline.pt"))
    return components, captured


@pytest.mark.parametrize("mode", ["research", "pilot"])
def test_prediction_uses_saved_cutoff_and_shared_text_assembly(mode):
    components, captured = components_for_test(mode)
    ocr = FakeOCR("OCR content")
    result = analysis.analyze_image(Image.new("RGB", (8, 8)), "caption", mode,
                                    ocr_loader=lambda: ocr, model_loader=lambda: components)
    assert result["status"] == "complete"
    assert result["threshold"] == 0.8
    assert result["class_scores"]["fake"] == pytest.approx(0.7310586)
    assert result["label"] == "real"  # largest score is fake, but below validation cutoff
    assert captured["text"] == "OCR content [SEP] caption"
    assert result["model_status"] == ("compatible_checkpoint" if mode == "research"
                                       else "unvalidated_internal_pilot")
    assert bool(result["warnings"]) is (mode == "pilot")
    assert not ocr.paths[0].exists()
    json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("logits", [[[float("nan"), 1.0]], [[float("inf"), 1.0]], [[1.0, 2.0, 3.0]]])
def test_nonfinite_and_wrong_class_outputs_never_become_predictions(logits):
    components, _ = components_for_test("pilot", logits=logits)
    result = analysis.analyze_image(Image.new("RGB", (8, 8)), mode="pilot",
                                    ocr_loader=lambda: FakeOCR(), model_loader=lambda: components)
    assert result["status"] == "error"
    assert result["error"]["code"] == "invalid_model_output"
    assert result["raw_ocr"]
    assert result["class_scores"] is result["label"] is None
    json.dumps(result, allow_nan=False)


def test_model_failure_preserves_ocr_without_exposing_exception_text():
    components, _ = components_for_test("research", fail=True)
    result = analysis.analyze_image(Image.new("RGB", (8, 8)), mode="research",
                                    ocr_loader=lambda: FakeOCR(), model_loader=lambda: components)
    assert result["status"] == "error" and result["raw_ocr"]
    assert result["class_scores"] is result["label"] is None
    assert "PRIVATE INPUT" not in json.dumps(result)


@pytest.mark.parametrize("throws", [False, True])
def test_ocr_failure_cleans_temp_and_prevents_classifier(throws):
    paths = []

    class OCR:
        def read(self, path):
            paths.append(Path(path))
            if throws:
                raise RuntimeError("private backend error")
            return OCRResult(error="private backend error")

    def forbidden():
        pytest.fail("OCR failure must prevent classifier initialization")

    result = analysis.analyze_image(Image.new("RGB", (8, 8)), mode="pilot",
                                    ocr_loader=lambda: OCR(), model_loader=forbidden)
    assert result["status"] == "error"
    assert result["class_scores"] is result["label"] is None
    assert paths and all(not path.exists() for path in paths)
    assert "private backend error" not in json.dumps(result)


def test_empty_ocr_and_disclosures_do_not_assign_truth():
    result = analysis.analyze_image(Image.new("RGB", (8, 8)), "subject to market risk; SEBI",
                                    ocr_loader=lambda: FakeOCR(""))
    assert result["status"] == "complete" and result["warnings"]
    assert result["label"] is None
    assert {cue["id"] for cue in result["indicators"]} == {"risk_notice", "authority"}
    assert all("severity" not in cue for cue in result["indicators"])


def test_invalid_image_returns_safe_error_without_loading_resources():
    def forbidden():
        pytest.fail("invalid image must not initialize resources")

    result = analysis.analyze_image(None, ocr_loader=forbidden, model_loader=forbidden)
    assert result["status"] == "error" and result["error"]["code"] == "input_failed"
    assert result["label"] is None
    json.dumps(result, allow_nan=False)
