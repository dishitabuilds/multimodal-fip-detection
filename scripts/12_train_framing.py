"""Train independent framing labels using reviewed promotion annotations.

See docs/FRAMING_LABELS.md before creating annotations. Binary fake/real labels
and automatic audit flags are not substitutes for the required human review.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402


def main(argv=None):
    from fipd.training.framing import FRAMING_LABELS, run_framing_experiment

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=Path("data/reviewed/framing.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/framing"))
    parser.add_argument("--labels", nargs="+", choices=FRAMING_LABELS)
    parser.add_argument("--C", dest="regularization", nargs="+", type=float, default=[0.1, 1, 10])
    parser.add_argument("--max-features", type=int, default=10000)
    parser.add_argument("--experiment-id")
    args = parser.parse_args(argv)
    try:
        run_dir, report = run_framing_experiment(
            args.records, args.output_dir, labels=args.labels,
            regularization=args.regularization, max_features=args.max_features,
            experiment_id=args.experiment_id)
        print(f"Completed reviewed-data framing run: {run_dir.resolve()}")
        print(f"Selected C using validation: {report['selected_C']}")
        print("Metrics apply to this supplied dataset. No fraud or stock-return verdict is produced.")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Framing training stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
