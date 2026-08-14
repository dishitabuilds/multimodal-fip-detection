"""Stage 6 — training loop, metrics and the ablation runner.

`metrics` is numpy-only and imports here eagerly; the training loop pulls torch
in lazily so that scoring a saved set of predictions never requires it.
"""

from .metrics import (
    ConfusionMatrix,
    Metrics,
    SeedAggregate,
    ablation_table,
    aggregate,
    best_threshold,
    confusion,
    evaluate,
    macro_f1,
    pr_auc,
    pr_curve,
    roc_auc,
)

__all__ = [
    "ConfusionMatrix",
    "Metrics",
    "SeedAggregate",
    "ablation_table",
    "aggregate",
    "best_threshold",
    "confusion",
    "evaluate",
    "macro_f1",
    "pr_auc",
    "pr_curve",
    "roc_auc",
]
