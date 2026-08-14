"""Evaluation metrics for an imbalanced binary fraud detector.

Accuracy is deliberately not a headline metric here. On a corpus that is mostly
one class, a model that answers "fake" every time scores well on accuracy and
is worth nothing; precision, recall, F1 and PR-AUC all expose that immediately.
The majority-class baseline is reported alongside every result for exactly that
reason — a number only means something next to what you would get for free.

Implemented in numpy rather than pulled from sklearn so the arithmetic is
inspectable and testable with no heavyweight import. The test suite cross-checks
these against sklearn when it happens to be installed.

Convention: label 1 is the positive class (`fake`), label 0 is `real`.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Sequence

import numpy as np


# ---------------------------------------------------------------------------


@dataclass
class ConfusionMatrix:
    tn: int = 0
    fp: int = 0
    fn: int = 0
    tp: int = 0

    @property
    def total(self) -> int:
        return self.tn + self.fp + self.fn + self.tp

    def __str__(self) -> str:
        return (
            f"{'':<12}{'pred real':>11}{'pred fake':>11}\n"
            f"{'true real':<12}{self.tn:>11}{self.fp:>11}\n"
            f"{'true fake':<12}{self.fn:>11}{self.tp:>11}"
        )


@dataclass
class Metrics:
    """One arm's scores at one decision threshold."""

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    macro_f1: float = 0.0
    pr_auc: float = 0.0
    roc_auc: float = 0.0
    accuracy: float = 0.0           # reported for completeness, never headline
    majority_baseline: float = 0.0  # what predicting the common class would give
    threshold: float = 0.5
    n: int = 0
    n_positive: int = 0
    confusion: ConfusionMatrix = field(default_factory=ConfusionMatrix)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"P {self.precision:.3f}  R {self.recall:.3f}  F1 {self.f1:.3f}  "
            f"macro-F1 {self.macro_f1:.3f}  PR-AUC {self.pr_auc:.3f}  "
            f"(acc {self.accuracy:.3f} vs majority {self.majority_baseline:.3f})"
        )


# ---------------------------------------------------------------------------


def _as_arrays(y_true: Sequence, y_score: Sequence) -> tuple[np.ndarray, np.ndarray]:
    yt = np.asarray(y_true, dtype=int).ravel()
    ys = np.asarray(y_score, dtype=float).ravel()
    if yt.shape != ys.shape:
        raise ValueError(f"length mismatch: {yt.shape[0]} labels, {ys.shape[0]} scores")
    if yt.size == 0:
        raise ValueError("no samples to score")
    bad = set(np.unique(yt)) - {0, 1}
    if bad:
        raise ValueError(f"labels must be 0/1, found {sorted(bad)}")
    return yt, ys


def confusion(y_true: Sequence, y_pred: Sequence) -> ConfusionMatrix:
    yt = np.asarray(y_true, dtype=int).ravel()
    yp = np.asarray(y_pred, dtype=int).ravel()
    return ConfusionMatrix(
        tn=int(np.sum((yt == 0) & (yp == 0))),
        fp=int(np.sum((yt == 0) & (yp == 1))),
        fn=int(np.sum((yt == 1) & (yp == 0))),
        tp=int(np.sum((yt == 1) & (yp == 1))),
    )


def precision_recall_f1(cm: ConfusionMatrix) -> tuple[float, float, float]:
    """Positive-class scores. A denominator of zero yields 0.0, not NaN —
    a model that never predicts `fake` has no precision, and propagating NaN
    into a mean over seeds silently destroys the whole column."""
    prec = cm.tp / (cm.tp + cm.fp) if (cm.tp + cm.fp) else 0.0
    rec = cm.tp / (cm.tp + cm.fn) if (cm.tp + cm.fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


def macro_f1(cm: ConfusionMatrix) -> float:
    """Mean of the two per-class F1s.

    This is the number the budget's retention row uses, because it weights the
    rare class equally instead of letting the common one carry the score.
    """
    _, _, f1_pos = precision_recall_f1(cm)
    flipped = ConfusionMatrix(tn=cm.tp, fp=cm.fn, fn=cm.fp, tp=cm.tn)
    _, _, f1_neg = precision_recall_f1(flipped)
    return (f1_pos + f1_neg) / 2


def pr_curve(y_true: Sequence, y_score: Sequence) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Precision/recall at every threshold the scores actually distinguish.

    Returns (precision, recall, thresholds), recall ascending. A detector that
    flags everything has perfect recall and no value, which is visible here and
    invisible in any single-threshold number.
    """
    yt, ys = _as_arrays(y_true, y_score)
    order = np.argsort(-ys, kind="mergesort")
    yt, ys = yt[order], ys[order]

    tp = np.cumsum(yt)
    fp = np.cumsum(1 - yt)
    total_pos = int(yt.sum())

    # One point per distinct score; ties must be resolved together or both
    # precision and recall get credit for a split that never happens.
    last = np.r_[np.flatnonzero(np.diff(ys)), ys.size - 1]
    tp, fp, thresholds = tp[last], fp[last], ys[last]

    precision = np.divide(tp, tp + fp, out=np.ones_like(tp, dtype=float),
                          where=(tp + fp) > 0)
    recall = tp / total_pos if total_pos else np.zeros_like(tp, dtype=float)
    return precision, recall, thresholds


def pr_auc(y_true: Sequence, y_score: Sequence) -> float:
    """Average precision — the step-wise area under the PR curve.

    Trapezoidal integration over a PR curve is optimistically biased, so this
    uses the sum of precision weighted by the change in recall, which is what
    sklearn's `average_precision_score` reports.
    """
    yt, _ = _as_arrays(y_true, y_score)
    if yt.sum() == 0 or yt.sum() == yt.size:
        return 0.0  # undefined with one class present; do not fabricate a number
    precision, recall, _ = pr_curve(y_true, y_score)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y_true: Sequence, y_score: Sequence) -> float:
    """Rank-based AUC via the Mann-Whitney statistic, ties averaged."""
    yt, ys = _as_arrays(y_true, y_score)
    n_pos, n_neg = int(yt.sum()), int((1 - yt).sum())
    if not n_pos or not n_neg:
        return 0.0

    order = np.argsort(ys, kind="mergesort")
    ranks = np.empty(ys.size, dtype=float)
    ranks[order] = np.arange(1, ys.size + 1, dtype=float)

    # Average ranks within tied score groups.
    sorted_scores = ys[order]
    i = 0
    while i < sorted_scores.size:
        j = i
        while j + 1 < sorted_scores.size and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + j + 2) / 2.0
        i = j + 1

    return float((ranks[yt == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def evaluate(y_true: Sequence, y_score: Sequence, threshold: float = 0.5) -> Metrics:
    """Full metric set at one threshold, plus the baseline it must beat."""
    yt, ys = _as_arrays(y_true, y_score)
    y_pred = (ys >= threshold).astype(int)
    cm = confusion(yt, y_pred)
    prec, rec, f1 = precision_recall_f1(cm)

    n_pos = int(yt.sum())
    majority = max(n_pos, yt.size - n_pos) / yt.size

    return Metrics(
        precision=prec,
        recall=rec,
        f1=f1,
        macro_f1=macro_f1(cm),
        pr_auc=pr_auc(yt, ys),
        roc_auc=roc_auc(yt, ys),
        accuracy=float(np.mean(y_pred == yt)),
        majority_baseline=majority,
        threshold=threshold,
        n=int(yt.size),
        n_positive=n_pos,
        confusion=cm,
    )


def best_threshold(y_true: Sequence, y_score: Sequence, metric: str = "f1") -> tuple[float, float]:
    """Threshold maximising `f1` or `macro_f1`, and the score there.

    Pick this on validation, never on test. Choosing a threshold on the test set
    is a quiet way of fitting to it.
    """
    if metric not in {"f1", "macro_f1"}:
        raise ValueError(f"metric must be 'f1' or 'macro_f1', got {metric!r}")
    _, ys = _as_arrays(y_true, y_score)

    best_t, best_v = 0.5, -1.0
    for t in np.unique(ys):
        m = evaluate(y_true, ys, threshold=float(t))
        v = m.f1 if metric == "f1" else m.macro_f1
        if v > best_v:
            best_t, best_v = float(t), v
    return best_t, best_v


# ---------------------------------------------------------------------------
# Aggregation across seeds
# ---------------------------------------------------------------------------


@dataclass
class SeedAggregate:
    """mean ± std of one metric over several seeds. A 1% gain on one seed
    is noise, so nothing gets reported without this."""

    metric: str
    mean: float
    std: float
    n_seeds: int
    values: list[float] = field(default_factory=list)

    def __str__(self) -> str:
        return f"{self.mean:.3f} ± {self.std:.3f} (n={self.n_seeds})"


def aggregate(runs: Sequence[Metrics], metric: str = "macro_f1") -> SeedAggregate:
    """Collapse per-seed Metrics into mean ± std for one metric."""
    if not runs:
        raise ValueError("no runs to aggregate")
    values = [float(getattr(r, metric)) for r in runs]
    return SeedAggregate(
        metric=metric,
        mean=float(np.mean(values)),
        std=float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
        n_seeds=len(values),
        values=values,
    )


def ablation_table(
    arms: dict[str, Sequence[Metrics]],
    metric: str = "macro_f1",
    reference: str = "text_only",
) -> str:
    """The table Dr. Surati asked for: each arm, and its delta over the baseline.

    The delta column carries the sign and magnitude, because "fusion helps" is
    the claim under test and three separate numbers do not state it.
    """
    aggs = {name: aggregate(runs, metric) for name, runs in arms.items() if runs}
    if not aggs:
        return "no runs to tabulate"

    base = aggs.get(reference)
    w = 62
    lines = [
        "=" * w,
        f"ABLATION — {metric}" + (f"   (delta vs {reference})" if base else ""),
        "=" * w,
        f"{'arm':<16}{'mean':>9}{'std':>9}{'seeds':>7}{'delta':>12}",
        "-" * w,
    ]
    for name in sorted(aggs, key=lambda k: -aggs[k].mean):
        a = aggs[name]
        if base is None or name == reference:
            delta = "  —"
        else:
            d = a.mean - base.mean
            delta = f"{d:+.3f}"
        lines.append(f"{name:<16}{a.mean:>9.3f}{a.std:>9.3f}{a.n_seeds:>7}{delta:>12}")
    lines.append("-" * w)

    if base and "fused" in aggs:
        best_single = max((a.mean for n, a in aggs.items() if n != "fused"), default=None)
        if best_single is not None:
            gap = aggs["fused"].mean - best_single
            spread = max(aggs["fused"].std, 0.0)
            verdict = "fusion helps" if gap > spread else "within seed noise"
            lines.append(f"fused - best single modality: {gap:+.3f}   ({verdict})")
    lines.append("=" * w)
    return "\n".join(lines)
