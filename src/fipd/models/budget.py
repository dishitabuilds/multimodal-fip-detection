"""The lightweight budget: what the word in the project title actually means.

"Lightweight" is a claim, and a claim with no number attached cannot be wrong,
which makes it worthless in a report. This module turns it into six measurable
rows with pass/fail thresholds, loaded from `configs/model.yaml`:

    params · disk size · p50 latency · p95 latency · peak RAM · F1 retention

The last row is the one that stops the budget being gamed. Without it, a model
that always answers "fake" in 3 ms passes every other target.

Nothing here imports torch at module level. The arithmetic, the report and the
pass/fail logic are plain Python so they can be tested — and the budget argued
about — before any model exists.
"""

from __future__ import annotations

import json
import logging
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Sequence

log = logging.getLogger(__name__)

DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "model.yaml"


# ---------------------------------------------------------------------------
# The targets
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Ceilings the deployed model must sit under. See configs/model.yaml."""

    max_params_m: float = 50.0
    max_disk_mb: float = 100.0
    max_latency_p50_ms: float = 100.0
    max_latency_p95_ms: float = 250.0
    max_peak_ram_mb: float = 1024.0
    min_f1_retention: float = 0.95

    @classmethod
    def from_config(cls, path: str | Path | None = None) -> "Budget":
        import yaml

        cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text(encoding="utf-8"))
        known = {f for f in cls.__dataclass_fields__}
        return cls(**{k: float(v) for k, v in (cfg.get("budget") or {}).items()
                      if k in known})


@dataclass
class Measurement:
    """What a given model actually does. `None` means "not measured yet"."""

    name: str = "model"
    params_m: float | None = None
    disk_mb: float | None = None
    latency_p50_ms: float | None = None
    latency_p95_ms: float | None = None
    peak_ram_mb: float | None = None
    f1: float | None = None
    teacher_f1: float | None = None
    conditions: dict[str, Any] = field(default_factory=dict)

    @property
    def f1_retention(self) -> float | None:
        """Student F1 as a fraction of the teacher's. None if either is absent."""
        if self.f1 is None or not self.teacher_f1:
            return None
        return self.f1 / self.teacher_f1


@dataclass
class Check:
    """One row of the budget table."""

    name: str
    measured: float | None
    limit: float
    unit: str
    #: "max" = measured must not exceed limit; "min" = must not fall below
    direction: str = "max"

    @property
    def status(self) -> str:
        if self.measured is None:
            return "not measured"
        if self.direction == "max":
            return "PASS" if self.measured <= self.limit else "OVER"
        return "PASS" if self.measured >= self.limit else "UNDER"

    @property
    def ok(self) -> bool:
        return self.status == "PASS"

    @property
    def headroom(self) -> float | None:
        """Signed slack against the limit, in the check's own unit."""
        if self.measured is None:
            return None
        return (self.limit - self.measured) if self.direction == "max" \
            else (self.measured - self.limit)


def check_budget(measurement: Measurement, budget: Budget) -> list[Check]:
    """Score a measurement against the budget, one Check per row."""
    return [
        Check("parameters", measurement.params_m, budget.max_params_m, "M"),
        Check("disk size", measurement.disk_mb, budget.max_disk_mb, "MB"),
        Check("latency p50", measurement.latency_p50_ms, budget.max_latency_p50_ms, "ms"),
        Check("latency p95", measurement.latency_p95_ms, budget.max_latency_p95_ms, "ms"),
        Check("peak RAM", measurement.peak_ram_mb, budget.max_peak_ram_mb, "MB"),
        Check("F1 retention", measurement.f1_retention, budget.min_f1_retention,
              "x teacher", direction="min"),
    ]


def format_report(measurement: Measurement, budget: Budget) -> str:
    """The table that goes in the report. Failures are shown, not hidden —
    a missed target is the size/latency/accuracy tradeoff, which is a result."""
    checks = check_budget(measurement, budget)
    w = 66
    lines = [
        "=" * w,
        f"LIGHTWEIGHT BUDGET — {measurement.name}",
        "=" * w,
        f"{'metric':<16}{'measured':>12}{'target':>12}{'headroom':>12}{'':>4}status",
        "-" * w,
    ]
    for c in checks:
        meas = "  —" if c.measured is None else f"{c.measured:.4g}"
        head = "  —" if c.headroom is None else f"{c.headroom:+.4g}"
        arrow = "<=" if c.direction == "max" else ">="
        lines.append(
            f"{c.name:<16}{meas:>12}{arrow + ' ' + f'{c.limit:.4g}':>12}"
            f"{head:>12}    {c.status}"
        )
    lines.append("-" * w)

    measured = [c for c in checks if c.measured is not None]
    failed = [c for c in measured if not c.ok]
    missing = [c for c in checks if c.measured is None]

    if not measured:
        lines.append("nothing measured yet")
    elif failed:
        lines.append(f"OVER BUDGET on {len(failed)}/{len(measured)}: "
                     + ", ".join(c.name for c in failed))
    else:
        lines.append(f"within budget on all {len(measured)} measured rows")
    if missing:
        lines.append("not yet measured: " + ", ".join(c.name for c in missing))

    if measurement.conditions:
        lines += ["", "measured under:"]
        lines += [f"  {k}: {v}" for k, v in sorted(measurement.conditions.items())]
    lines.append("=" * w)
    return "\n".join(lines)


def save_report(measurement: Measurement, budget: Budget, path: str | Path) -> Path:
    """Write the machine-readable form next to the printed table."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "measurement": asdict(measurement),
        "budget": asdict(budget),
        "checks": [
            {"name": c.name, "measured": c.measured, "limit": c.limit,
             "unit": c.unit, "direction": c.direction, "status": c.status}
            for c in check_budget(measurement, budget)
        ],
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------


def measure_latency(
    fn: Callable[[], Any],
    warmup_runs: int = 10,
    timed_runs: int = 100,
) -> dict[str, float]:
    """Time `fn` and return p50/p95/mean/std/min/max in milliseconds.

    Warm-up runs are discarded: the first calls into a freshly loaded model pay
    lazy kernel selection, allocator growth and cold caches, and including them
    reports a number no user ever experiences twice.

    `perf_counter` rather than `time()` — on Windows the latter has ~16 ms
    granularity, which is the same order as the thing being measured.
    """
    if timed_runs < 1:
        raise ValueError("timed_runs must be >= 1")

    for _ in range(max(0, warmup_runs)):
        fn()

    samples: list[float] = []
    for _ in range(timed_runs):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)

    samples.sort()
    return {
        "p50_ms": statistics.median(samples),
        "p95_ms": percentile(samples, 95),
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.pstdev(samples) if len(samples) > 1 else 0.0,
        "min_ms": samples[0],
        "max_ms": samples[-1],
        "runs": float(timed_runs),
    }


def percentile(sorted_samples: Sequence[float], pct: float) -> float:
    """Linear-interpolated percentile of an already-sorted sequence."""
    if not sorted_samples:
        raise ValueError("no samples")
    if len(sorted_samples) == 1:
        return float(sorted_samples[0])
    pos = (len(sorted_samples) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(sorted_samples) - 1)
    frac = pos - lo
    return float(sorted_samples[lo] * (1 - frac) + sorted_samples[hi] * frac)


def count_parameters(model, trainable_only: bool = False) -> int:
    """Parameter count for a torch module. Frozen backbones still occupy disk
    and still have to be loaded, so the default counts them."""
    return sum(p.numel() for p in model.parameters()
               if p.requires_grad or not trainable_only)


def state_dict_size_mb(model) -> float:
    """Size of the weights in memory, in MB, from dtype and shape.

    This is the fp32/fp16 footprint. The INT8 figure the budget cares about
    comes from the exported artefact on disk — use `file_size_mb` for that,
    because quantisation changes both dtype and layout.
    """
    total_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    total_bytes += sum(b.numel() * b.element_size() for b in model.buffers())
    return total_bytes / (1024 * 1024)


def file_size_mb(path: str | Path) -> float:
    """Size of an exported model file on disk, in MB."""
    return Path(path).stat().st_size / (1024 * 1024)


def peak_ram_mb() -> float | None:
    """Peak resident set size of this process, in MB, or None if unavailable.

    psutil gives the current RSS on every platform; the true peak needs
    per-platform APIs, so this is sampled at the end of a measurement run and
    is a lower bound. Returning None rather than guessing keeps an unmeasured
    row visibly unmeasured in the report.
    """
    try:
        import psutil
    except ImportError:
        log.info("psutil not installed — peak RAM will show as not measured")
        return None
    try:
        return psutil.Process().memory_info().rss / (1024 * 1024)
    except Exception as e:  # noqa: BLE001 - a metric must never kill a run
        log.warning("could not read RSS: %s", e)
        return None
