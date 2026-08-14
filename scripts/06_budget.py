#!/usr/bin/env python3
"""Stage 06 — measure a model against the lightweight budget.

The project title claims the model is "lightweight". This is the test of that
claim. Targets live in configs/model.yaml; this script measures a model and
prints the pass/fail table that goes in the report.

    # what are we aiming at? (no torch needed, no model needed)
    python scripts/06_budget.py --targets-only

    # measure an architecture before training it, random weights, no downloads
    python scripts/06_budget.py --arm fused --variant student --from-scratch

    # measure the real thing, and score it against the teacher it distilled from
    python scripts/06_budget.py --arm fused --variant student \\
        --checkpoint checkpoints/fused_seed42.pt --f1 0.81 --teacher-f1 0.84

Output: data/processed/budget_<name>.json plus the printed table.

A row that fails is a result, not an embarrassment — it is the size/latency/
accuracy tradeoff that Phase 4 of docs/CHECKLIST.md asks you to report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.models.budget import (  # noqa: E402
    Budget,
    Measurement,
    count_parameters,
    file_size_mb,
    format_report,
    measure_latency,
    peak_ram_mb,
    save_report,
    state_dict_size_mb,
)
from fipd.models.config import load_model_config  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to model.yaml")
    ap.add_argument("--arm", default="fused", choices=["text_only", "image_only", "fused"])
    ap.add_argument("--variant", default="student", choices=["student", "teacher"])
    ap.add_argument("--from-scratch", action="store_true",
                    help="random weights instead of downloading pretrained ones")
    ap.add_argument("--checkpoint", default=None, help="measure this .pt/.onnx instead")
    ap.add_argument("--targets-only", action="store_true",
                    help="print the budget and exit; no torch required")
    ap.add_argument("--f1", type=float, default=None, help="this model's macro-F1")
    ap.add_argument("--teacher-f1", type=float, default=None,
                    help="teacher macro-F1, for the retention row")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    log = setup(args.verbose)
    cfg = load_model_config(args.config)
    budget = Budget.from_config(args.config)
    m = cfg.measurement

    name = f"{args.variant}/{args.arm}"
    banner(f"Stage 06 — lightweight budget ({name})")

    measurement = Measurement(
        name=name,
        f1=args.f1,
        teacher_f1=args.teacher_f1,
        conditions={
            "batch_size": m.batch_size,
            "threads": m.threads,
            "image_size": m.image_size,
            "max_text_tokens": m.max_text_tokens,
            "precision": m.precision,
            "timed_runs": m.timed_runs,
            "reference_machine": m.reference_machine,
        },
    )

    if args.targets_only:
        print("\n" + format_report(measurement, budget))
        print("\nNothing measured — these are the targets. Build a model and\n"
              "re-run without --targets-only to fill the table in.\n")
        return 0

    # ---- build and measure ------------------------------------------------
    try:
        import torch

        from fipd.models.arms import build_arm
    except ImportError as e:
        log.error("%s", e)
        log.error('install the model extras: pip install -e ".[model]"')
        log.error("or run with --targets-only to see the budget alone")
        return 1

    torch.set_num_threads(m.threads)
    arch = getattr(cfg, args.variant)
    log.info("building %s arm of the %s (%s / %s)", args.arm, args.variant,
             arch.text_encoder, arch.image_encoder)

    model = build_arm(args.arm, arch, from_scratch=args.from_scratch)
    model.eval()

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(ckpt.get("state_dict", ckpt))
        measurement.disk_mb = file_size_mb(args.checkpoint)
        log.info("loaded checkpoint %s", args.checkpoint)
    else:
        measurement.disk_mb = state_dict_size_mb(model)
        log.info("no checkpoint given — disk size is the fp32 weight footprint, "
                 "not the INT8 export the budget targets")

    measurement.params_m = count_parameters(model) / 1e6

    # One synthetic sample, shaped exactly as the stated conditions say.
    ids = torch.randint(0, 1000, (m.batch_size, m.max_text_tokens))
    mask = torch.ones_like(ids)
    pixels = torch.randn(m.batch_size, 3, m.image_size, m.image_size)

    kw = {}
    if args.arm in ("text_only", "fused"):
        kw.update(input_ids=ids, attention_mask=mask)
    if args.arm in ("image_only", "fused"):
        kw.update(pixel_values=pixels)

    def run_once():
        with torch.no_grad():
            model(**kw)

    log.info("timing: %d warm-up + %d timed runs on %d threads",
             m.warmup_runs, m.timed_runs, m.threads)
    timing = measure_latency(run_once, m.warmup_runs, m.timed_runs)
    measurement.latency_p50_ms = timing["p50_ms"]
    measurement.latency_p95_ms = timing["p95_ms"]
    measurement.peak_ram_mb = peak_ram_mb()
    measurement.conditions["torch"] = torch.__version__
    measurement.conditions["weights"] = "random" if args.from_scratch else "pretrained"

    print("\n" + format_report(measurement, budget))

    out = save_report(measurement, budget,
                      Path(args.out_dir) / f"budget_{args.variant}_{args.arm}.json")
    print(f"\nwrote {out}")
    if measurement.f1 is None:
        print("\nNote: pass --f1 and --teacher-f1 to fill the retention row. Without\n"
              "it the budget can be passed by a model that is fast and useless.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
