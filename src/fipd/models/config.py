"""Typed access to configs/model.yaml.

Same principle as configs/sources.yaml: every tunable lives in the YAML, and
code reads it through here rather than hard-coding a number that then has to be
hunted down when a result needs explaining. No torch import — a config should
be loadable and printable on a machine that cannot train anything.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "model.yaml"

#: The three ablation arms. `fused` must beat the single-modality arms on
#: image-only-claim posts or the project's central thesis is not demonstrated.
ARMS = ("text_only", "image_only", "fused")


@dataclass
class ArchConfig:
    """One model variant — the teacher, or the student it distils into."""

    text_encoder: str = "google/muril-base-cased"
    image_encoder: str = "google/vit-base-patch16-224"
    fusion: str = "cross_attention"
    hidden_dim: int = 768
    fusion_heads: int = 8
    fusion_layers: int = 2
    dropout: float = 0.1
    num_labels: int = 2


@dataclass
class TextConfig:
    """How the caption and the OCR'd in-image text become one string.

    The separator is deliberate, not cosmetic: caption and in-image text are
    different kinds of evidence — the caption is usually the benign half — and
    an explicit boundary token lets the encoder represent that. Changing this
    changes what the text arm sees, so a result measured under one setting
    cannot be compared with a result measured under another.
    """

    fields: list[str] = field(default_factory=lambda: ["claim_text", "title", "ocr_text"])
    separator: str = " [SEP] "
    max_chars: int = 2000
    lowercase: bool = False
    transliterate: bool = True


@dataclass
class TrainConfig:
    seeds: list[int] = field(default_factory=lambda: [13, 42, 1337])
    epochs: int = 8
    batch_size: int = 16
    lr: float = 2e-5
    head_lr: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    early_stopping_patience: int = 3
    monitor: str = "val_macro_f1"
    class_weighting: str = "balanced"
    mixed_precision: bool = True
    num_workers: int = 2


@dataclass
class MeasurementConfig:
    """Conditions the latency numbers are only meaningful under."""

    batch_size: int = 1
    threads: int = 4
    image_size: int = 224
    max_text_tokens: int = 128
    warmup_runs: int = 10
    timed_runs: int = 100
    precision: str = "int8"
    reference_machine: str = ""


@dataclass
class ModelConfig:
    teacher: ArchConfig = field(default_factory=ArchConfig)
    student: ArchConfig = field(default_factory=ArchConfig)
    arms: list[str] = field(default_factory=lambda: list(ARMS))
    text: TextConfig = field(default_factory=TextConfig)
    training: TrainConfig = field(default_factory=TrainConfig)
    measurement: MeasurementConfig = field(default_factory=MeasurementConfig)
    evaluation: dict[str, Any] = field(default_factory=dict)
    distillation: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)


def _build(cls, data: dict | None):
    """Instantiate a config dataclass, ignoring keys it does not declare.

    Tolerant on purpose: an unrecognised key in the YAML is a comment or a
    setting for a later phase, and neither should stop a run.
    """
    known = set(cls.__dataclass_fields__)
    return cls(**{k: v for k, v in (data or {}).items() if k in known})


def load_model_config(path: str | Path | None = None) -> ModelConfig:
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8")) or {}
    model = raw.get("model") or {}

    arms = list(model.get("arms") or ARMS)
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise ValueError(f"unknown ablation arm(s) {unknown}; expected any of {list(ARMS)}")

    return ModelConfig(
        teacher=_build(ArchConfig, model.get("teacher")),
        student=_build(ArchConfig, model.get("student")),
        arms=arms,
        text=_build(TextConfig, raw.get("text")),
        training=_build(TrainConfig, raw.get("training")),
        measurement=_build(MeasurementConfig, raw.get("measurement")),
        evaluation=raw.get("evaluation") or {},
        distillation=raw.get("distillation") or {},
        raw=raw,
    )
