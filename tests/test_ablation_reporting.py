"""A single seed and a large negative fusion gap cannot become evidence of benefit."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from fipd.training.metrics import Metrics, ablation_table


def runs(*values):
    return [Metrics(macro_f1=value) for value in values]


def test_single_seed_does_not_estimate_seed_noise():
    report = ablation_table({"text_only": runs(1.0), "fused": runs(0.667)})
    assert "single seed; uncertainty unmeasured" in report
    assert "within seed noise" not in report


def test_large_negative_gap_is_not_called_within_noise():
    report = ablation_table({"text_only": runs(0.85, 0.86), "fused": runs(0.20, 0.21)})
    assert "below best single modality beyond observed seed spread" in report
    assert "not a significance test" in report


def test_both_modalities_seed_spreads_contribute():
    report = ablation_table({"text_only": runs(0.3, 0.9), "fused": runs(0.60, 0.61)})
    assert "within observed seed spread" in report


def test_positive_gap_is_described_without_claiming_significance():
    report = ablation_table({"text_only": runs(0.5, 0.51), "fused": runs(0.90, 0.91)})
    assert "above best single modality beyond observed seed spread" in report
    assert "not a significance test" in report
