"""Validate a deployed checkpoint's input contract before loading encoders."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict

from ..datasets.examples import INPUT_PIPELINE_VERSION
from .config import ModelConfig


def validate_frontend_checkpoint(checkpoint: Mapping, cfg: ModelConfig) -> float:
    """Return its cutoff only when the student fused model matches this frontend.

    Matching tensor shapes do not prove matching tokenization/text assembly.
    Legacy metadata-free artifacts require verified re-export or retraining;
    their input contract is never silently guessed.
    """
    if not isinstance(checkpoint, Mapping) or not isinstance(checkpoint.get("state_dict"), Mapping):
        raise ValueError("Checkpoint must contain a state_dict and training provenance.")
    if checkpoint.get("arm") != "fused" or checkpoint.get("variant") != "student":
        raise ValueError("This frontend requires a student fused-arm checkpoint.")
    saved = checkpoint.get("config")
    if not isinstance(saved, Mapping) or saved.get("input_pipeline_version") != INPUT_PIPELINE_VERSION:
        raise ValueError("Checkpoint input-pipeline provenance is missing or incompatible. "
                         "Use a verified checkpoint from the current training runner.")
    if saved.get("architecture") != asdict(cfg.student):
        raise ValueError("Checkpoint architecture/tokenizer does not match the frontend configuration.")
    if saved.get("text") != asdict(cfg.text):
        raise ValueError("Checkpoint text preprocessing does not match the frontend configuration.")
    measurement = saved.get("measurement")
    if not isinstance(measurement, Mapping) or any(
        measurement.get(key) != getattr(cfg.measurement, key)
        for key in ("image_size", "max_text_tokens")
    ):
        raise ValueError("Checkpoint image size or token limit does not match the frontend configuration.")
    try:
        threshold = float(checkpoint["threshold"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Checkpoint must contain its validation-selected decision threshold.") from exc
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Checkpoint decision threshold must be between 0 and 1.")
    return threshold
