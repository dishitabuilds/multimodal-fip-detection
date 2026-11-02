"""Predict with a locally trained lab artifact and optional local OCR."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402


def main(argv=None):
    from fipd.models.classical import (
        DEFAULT_IMAGE_WEIGHTS, MobileNetFeatures, load_lab_artifact, predict_lab,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--ocr-text", default="")
    parser.add_argument("--caption", default="")
    parser.add_argument("--ocr", action="store_true", help="extract text with prepared local EasyOCR")
    parser.add_argument("--internal-pilot", action="store_true")
    parser.add_argument("--image-weights", type=Path, default=DEFAULT_IMAGE_WEIGHTS)
    args = parser.parse_args(argv)
    try:
        artifact = load_lab_artifact(args.artifact, allow_internal=args.internal_pilot)
        encoder, image = None, None
        text = args.ocr_text
        if args.image:
            from PIL import Image
            from fipd.models.classical import canonical_image

            with Image.open(args.image) as source:
                image = canonical_image(source)
        if args.ocr:
            if args.image is None:
                raise ValueError("--ocr requires --image")
            from fipd.frontend.analysis import load_ocr

            result = load_ocr(Path(__file__).resolve().parents[1]).read(args.image)
            if result.error:
                raise ValueError(f"OCR failed: {result.error}")
            text = result.text
        if artifact["arm"] != "text_only":
            encoder = MobileNetFeatures(args.image_weights)
        output = predict_lab(artifact, text, args.caption, image, encoder)
        print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=False))
        return 0
    except (ValueError, OSError, ImportError) as exc:
        print(f"Lab prediction stopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
