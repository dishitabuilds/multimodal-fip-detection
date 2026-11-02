"""Check preprocessing compatibility without importing or downloading encoders."""

import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fipd.datasets.examples import INPUT_PIPELINE_VERSION
from fipd.models.artifacts import validate_frontend_checkpoint
from fipd.models.config import load_model_config


@pytest.fixture
def artifact():
    cfg = load_model_config()
    ckpt = {"state_dict": {}, "arm": "fused", "variant": "student", "threshold": 0.37,
            "config": {"input_pipeline_version": INPUT_PIPELINE_VERSION,
                       "architecture": asdict(cfg.student), "text": asdict(cfg.text),
                       "measurement": asdict(cfg.measurement)}}
    return cfg, ckpt


def test_matching_artifact_keeps_validation_cutoff(artifact):
    cfg, ckpt = artifact
    assert validate_frontend_checkpoint(ckpt, cfg) == 0.37


@pytest.mark.parametrize("mutation", [
    lambda c: c.pop("config"),
    lambda c: c["config"].update(input_pipeline_version="old-all-images-ocr"),
    lambda c: c.update(variant="teacher"),
    lambda c: c["config"]["text"].update(fields=["ocr_text"]),
    lambda c: c["config"]["architecture"].update(text_encoder="other-same-shape-tokenizer"),
    lambda c: c["config"]["measurement"].update(max_text_tokens=256),
    lambda c: c["config"]["measurement"].update(image_size=128),
    lambda c: c.update(threshold=float("nan")),
    lambda c: c.pop("threshold"),
])
def test_incompatible_or_unverifiable_artifacts_fail_before_loading(artifact, mutation):
    cfg, ckpt = artifact
    original = copy.deepcopy(ckpt)
    mutation(ckpt)
    with pytest.raises(ValueError):
        validate_frontend_checkpoint(ckpt, cfg)
    assert cfg.student.text_encoder == original["config"]["architecture"]["text_encoder"]
