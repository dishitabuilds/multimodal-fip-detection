"""Train lab baselines or explicitly prepare pretrained image weights.

Examples (from the project root):
    python scripts/10_train_lab.py --arms text_only
    python scripts/10_train_lab.py --prepare-image-model
    python scripts/10_train_lab.py

Research runs require manually reviewed data and both classes in every split.
--internal-pilot permits provisional data solely for engineering demonstrations.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402


def main(argv=None):
    from fipd.models.classical import DEFAULT_IMAGE_WEIGHTS, MODEL_NAMES, prepare_image_weights
    from fipd.models.config import ARMS

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--records", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/lab.yaml"))
    parser.add_argument("--arms", nargs="+", choices=ARMS)
    parser.add_argument("--models", nargs="+", choices=MODEL_NAMES)
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--experiment-id")
    parser.add_argument("--internal-pilot", action="store_true")
    parser.add_argument("--image-weights", type=Path, default=DEFAULT_IMAGE_WEIGHTS)
    parser.add_argument("--prepare-image-model", action="store_true",
                        help="explicitly download the official 9.8 MB pretrained MobileNet checkpoint")
    args = parser.parse_args(argv)
    try:
        if args.prepare_image_model:
            print(prepare_image_weights(args.image_weights).resolve())
            return 0
        import yaml
        from fipd.training.lab import run_lab_experiment

        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("lab configuration must be an object")
        if args.internal_pilot:
            print("INTERNAL ENGINEERING PILOT: provisional data; metrics are unvalidated.", flush=True)
        run_dir, identity = run_lab_experiment(
            args.records or args.data_dir / "interim/finance.jsonl",
            args.manifest or args.data_dir / "processed/split_manifest.json",
            args.data_dir, args.out_dir or args.data_dir / "processed/lab", config,
            arms=args.arms, models=args.models, internal_pilot=args.internal_pilot,
            experiment_id=args.experiment_id, image_weights=args.image_weights)
        print(f"Completed {len(identity['results'])} model comparisons: {run_dir.resolve()}")
        print(f"Selected using validation data: {identity['recommended_artifact']}")
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Lab training stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
