"""Measure the local OCR-to-prediction path on CPU for a saved lab artifact."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import resource
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from PIL import Image

from fipd.frontend.analysis import analyze_image, load_ocr
from fipd.models.classical import MobileNetFeatures, load_lab_artifact, predict_lab, DEFAULT_IMAGE_WEIGHTS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")
    root = Path(__file__).resolve().parents[1]
    model = load_lab_artifact(args.artifact, allow_internal=True)
    encoder = MobileNetFeatures(root / DEFAULT_IMAGE_WEIGHTS) if model["arm"] != "text_only" else None
    ocr = load_ocr(root)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    samples = []
    for _ in range(args.runs):
        start = time.perf_counter()
        extracted = analyze_image(image, root=root, ocr_loader=lambda: ocr)
        if extracted["status"] != "complete":
            raise RuntimeError(extracted["error"])
        prediction = predict_lab(model, extracted["raw_ocr"], image=image, encoder=encoder)
        samples.append({"total_ms": (time.perf_counter() - start) * 1000,
                        "ocr_ms": extracted["durations_ms"]["ocr"], "label": prediction["label"]})
    times = sorted(item["total_ms"] for item in samples)
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak / (1024 * 1024) if platform.system() == "Darwin" else peak / 1024
    report = {"artifact": str(args.artifact), "image": str(args.image), "runs": args.runs,
              "machine": platform.platform(), "cpu": platform.processor(),
              "median_ms": statistics.median(times), "p95_ms": times[min(len(times)-1, int(.95 * len(times)))],
              "process_peak_ram_mb": peak_mb, "artifact_mb": args.artifact.stat().st_size / 1024**2,
              "image_weights_mb": (root / DEFAULT_IMAGE_WEIGHTS).stat().st_size / 1024**2 if encoder else 0,
              "samples": samples,
              "scope": "Warm models; image decoding, OCR, feature extraction and classification."
                       " Process peak includes loaded Python libraries and models."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ("median_ms", "p95_ms", "process_peak_ram_mb")}, indent=2))


if __name__ == "__main__":
    main()
