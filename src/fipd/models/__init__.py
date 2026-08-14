"""Stage 5 — encoders, cross-attention fusion, and the lightweight budget.

Import policy: everything reachable from this module must import without torch
installed. The budget and the config are needed on the data-collection machine;
the encoders and arms pull torch in lazily, when they are actually built.
"""

from .budget import (
    Budget,
    Check,
    Measurement,
    check_budget,
    format_report,
    measure_latency,
    save_report,
)
from .config import ARMS, ArchConfig, ModelConfig, TextConfig, load_model_config

__all__ = [
    "ARMS",
    "ArchConfig",
    "Budget",
    "Check",
    "Measurement",
    "ModelConfig",
    "TextConfig",
    "check_budget",
    "format_report",
    "load_model_config",
    "measure_latency",
    "save_report",
]
