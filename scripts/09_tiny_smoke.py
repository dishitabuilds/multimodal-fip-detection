#!/usr/bin/env python3
"""Prepare a download-free tiny random baseline; optionally run guarded Stage 07.

    python scripts/09_tiny_smoke.py
    python scripts/09_tiny_smoke.py --threads 4 --run-ablation -- \
        --records data/reviewed/records.jsonl --manifest data/processed/splits.json

The generated snapshots are UNTRAINED. This one-epoch/one-seed run is an internal
plumbing smoke. Stage 07 still requires valid two-class splits and passes all
normal cohort/leakage gates. No research accuracy is claimed here.
"""

from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.models.arms import build_arm  # noqa: E402
from fipd.models.config import load_model_config  # noqa: E402
from fipd.models.encoders import require_torch  # noqa: E402
from fipd.models.tiny import FORMAT_VERSION, ORIGIN, prepare_tiny_assets  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--threads", type=int, default=4, help="PyTorch CPU thread cap")
    parser.add_argument("--run-ablation", action="store_true", help="run Stage 07 after preparation")
    parser.add_argument("ablation_args", nargs=argparse.REMAINDER,
                        help="Stage 07 options following --; keep the smoke config/student")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be >= 1")
    forwarded = args.ablation_args
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if forwarded and not args.run_ablation:
        parser.error("Stage 07 options require --run-ablation")
    if any(arg.split("=", 1)[0] in {"--config", "--variant"} for arg in forwarded):
        parser.error("this wrapper uses configs/tiny-smoke.yaml and the student variant")

    root = Path(__file__).resolve().parents[1]
    if Path.cwd().resolve() != root:
        parser.error(f"run this smoke wrapper from the project root: {root}")
    try:
        torch = require_torch()
        torch.set_num_threads(args.threads)
        destination = root / "data/models/tiny-smoke"
        if destination.exists():
            manifest = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
            if (manifest.get("format_version") != FORMAT_VERSION or
                    manifest.get("weight_origin") != ORIGIN or manifest.get("seed") != 42):
                raise ValueError("existing assets do not match this smoke baseline")
            for relative in ("text/config.json", "text/tokenizer.json", "text/model.safetensors",
                             "image/config.json", "image/model.safetensors"):
                if not (destination / relative).is_file():
                    raise ValueError(f"incomplete smoke assets: {relative}")
        else:
            manifest = prepare_tiny_assets(destination, seed=42)

        print("UNTRAINED random tiny baseline — internal plumbing smoke only.")
        print(f"CPU threads: {args.threads}; assets: {destination}")
        print(f"Text encoder parameters: {manifest['text_parameters']:,}")
        print(f"Image encoder parameters: {manifest['image_parameters']:,}")
        config = load_model_config(root / "configs/tiny-smoke.yaml")
        for arm in config.arms:
            model = build_arm(arm, config.student)
            print(f"{arm} parameters: {sum(parameter.numel() for parameter in model.parameters()):,}")
            del model
    except (ImportError, OSError, ValueError) as error:
        print(f"Tiny smoke preparation failed: {error}", file=sys.stderr)
        return 1

    print("Config: configs/tiny-smoke.yaml (one epoch, batch 4, one seed 42)")
    if args.run_ablation:
        previous = sys.argv
        try:
            sys.argv = [str(root / "scripts/07_ablation.py"),
                        "--config", str(root / "configs/tiny-smoke.yaml"),
                        "--variant", "student", *forwarded]
            runpy.run_path(sys.argv[0], run_name="__main__")
        except SystemExit as result:
            return int(result.code or 0)
        finally:
            sys.argv = previous
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
