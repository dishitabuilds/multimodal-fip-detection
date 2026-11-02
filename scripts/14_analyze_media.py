#!/usr/bin/env python3
"""Prepare local speech weights explicitly, or analyze an audio/video upload."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.analysis.media import analyze_media  # noqa: E402
from fipd.enrichment.media import (  # noqa: E402
    DEFAULT_SPEECH_MODEL, SPEECH_REPOSITORY, MediaLimits, load_speech_model,
)

ROOT = Path(__file__).resolve().parents[1]


def prepare_model(path: Path):
    """Pin the resolved upstream commit and keep provenance with the weights."""
    from huggingface_hub import HfApi, snapshot_download

    revision = HfApi().model_info(SPEECH_REPOSITORY).sha
    snapshot_download(SPEECH_REPOSITORY, revision=revision, local_dir=path,
                      allow_patterns=["config.json", "model.bin", "tokenizer.json",
                                      "vocabulary.*", "preprocessor_config.json"],
                      max_workers=2)
    files = {}
    for file in path.iterdir():
        if file.is_file() and file.name != "preparation.json":
            with file.open("rb") as handle:
                digest = hashlib.sha256()
                for block in iter(lambda: handle.read(1024**2), b""):
                    digest.update(block)
            files[file.name] = digest.hexdigest()
    (path / "preparation.json").write_text(json.dumps({
        "repository": SPEECH_REPOSITORY, "revision": revision, "sha256": files,
        "prepared_at_utc": datetime.now(timezone.utc).isoformat(),
    }, indent=2) + "\n", encoding="utf-8")
    load_speech_model(ROOT, path)
    print(f"Prepared local CPU INT8 speech model: {path.resolve()}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", nargs="?", type=Path)
    parser.add_argument("--prepare-model", action="store_true", help="explicit model download")
    parser.add_argument("--model-dir", type=Path, default=ROOT / DEFAULT_SPEECH_MODEL)
    parser.add_argument("--output", type=Path, help="write a new JSON report; existing files are preserved")
    parser.add_argument("--caption", default="")
    parser.add_argument("--language", choices=["en", "hi"])
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--no-frames", action="store_true")
    parser.add_argument("--frame-interval", type=float, default=5.0)
    parser.add_argument("--artifact", type=Path, help="optional locally trained ML Lab artifact")
    parser.add_argument("--internal-pilot", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.prepare_model:
            prepare_model(args.model_dir)
            if args.file is None:
                return 0
        if args.file is None:
            parser.error("supply a media file or --prepare-model")
        limits = MediaLimits(frame_interval=args.frame_interval)
        if args.file.stat().st_size > limits.max_bytes:
            raise ValueError("Choose a file no larger than 50 MB.")
        artifact = encoder = None
        if args.artifact:
            from fipd.models.classical import DEFAULT_IMAGE_WEIGHTS, MobileNetFeatures, load_lab_artifact

            artifact = load_lab_artifact(args.artifact, allow_internal=args.internal_pilot)
            if artifact["arm"] != "text_only":
                encoder = MobileNetFeatures(ROOT / DEFAULT_IMAGE_WEIGHTS)
        report = analyze_media(
            args.file.read_bytes(), args.file.name, root=ROOT, caption=args.caption,
            language=args.language, speech=not args.no_speech, frames=not args.no_frames,
            limits=limits, artifact=artifact, encoder=encoder,
            speech_loader=lambda: load_speech_model(ROOT, args.model_dir),
            progress=lambda message: print(message, file=sys.stderr),
        )
        serialized = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8") as handle:
                handle.write(serialized)
            print(f"Report: {args.output.resolve()}")
        else:
            print(serialized)
        return 0 if report["status"] == "complete" else 2
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
