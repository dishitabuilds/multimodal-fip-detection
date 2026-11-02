"""Image-text inspection and explicitly selected checkpoint inference.

Importing this module never initializes OCR, torch or an encoder. The app owns
resource caching; zero-argument loader callbacks make that boundary explicit.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile
import time
from typing import Any, Callable, Sequence

from PIL import Image, ImageOps

from ..datasets.examples import INPUT_PIPELINE_VERSION, assemble_text
from ..models.config import ModelConfig, load_model_config
from ..schema.records import FactCheckRecord, ImageAsset

MODES = ("ocr", "research", "pilot")
RESEARCH_CHECKPOINT_RELATIVE = Path("data/checkpoints/fused_seed42.pt")
PILOT_CHECKPOINT_RELATIVE = Path(
    "data/checkpoints/student/internal-ai-pilot-20261025/fused_seed42.pt")
RESEARCH_CONFIG_RELATIVE = Path("configs/model.yaml")
PILOT_CONFIG_RELATIVE = Path("configs/tiny-smoke.yaml")
MODE_PURPOSES = {"ocr": "text_extraction_only", "research": "research_inference",
                 "pilot": "internal_smoke_only"}


class AnalysisError(RuntimeError):
    """A stable, user-facing operational failure without an exposed traceback."""

    def __init__(self, code: str, message: str, details: str = ""):
        super().__init__(message)
        self.code, self.message, self.details = code, message, details


@dataclass
class ClassifierComponents:
    model: Any
    tokenizer: Any
    transform: Any
    config: ModelConfig
    threshold: float
    mode: str
    purpose: str
    checkpoint_path: Path
    experiment_id: str | None = None


def canonical_image(image: Image.Image) -> Image.Image:
    """Match preview pixels: apply EXIF orientation and place alpha on white."""
    if not isinstance(image, Image.Image):
        raise ValueError("Choose a readable image before running analysis.")
    oriented = ImageOps.exif_transpose(image)
    if "A" in oriented.getbands() or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        normalized = Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba)
        return normalized.convert("RGB")
    return oriented.convert("RGB")


def input_fingerprint(image: Image.Image, caption: str = "", mode: str = "ocr") -> str:
    """Identify the actual pixels, exact caption, mode and assembly version."""
    if mode not in MODES or not isinstance(caption, str):
        raise ValueError("Use a supported analysis mode and a text caption.")
    normalized = canonical_image(image)
    header = json.dumps({"size": normalized.size, "caption": caption, "mode": mode,
                         "pipeline": INPUT_PIPELINE_VERSION}, sort_keys=True,
                        ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(header + b"\x00" + normalized.tobytes()).hexdigest()


def mode_runtime(root: Path, mode: str = "ocr") -> dict:
    """Read only mode paths/presence; this never loads a model or config."""
    if mode not in MODES:
        raise ValueError("Unknown analysis mode; choose OCR, research or pilot.")
    root = Path(root).resolve()
    checkpoint = root / (RESEARCH_CHECKPOINT_RELATIVE if mode == "research"
                         else PILOT_CHECKPOINT_RELATIVE) if mode != "ocr" else None
    config = root / (RESEARCH_CONFIG_RELATIVE if mode == "research"
                     else PILOT_CONFIG_RELATIVE) if mode != "ocr" else None
    return {"mode": mode, "purpose": MODE_PURPOSES[mode],
            "model_status": "not_loaded" if mode == "ocr" else "not_loaded_checkpoint_present"
            if checkpoint.is_file() else "unavailable",
            "checkpoint_path": str(checkpoint) if checkpoint else None,
            "config_path": str(config) if config else None,
            "checkpoint_available": checkpoint.is_file() if checkpoint else False,
            "unvalidated": mode == "pilot"}


def load_ocr(root: Path, languages: Sequence[str] = ("en", "hi")):
    """Initialize CPU OCR from the local cache, with downloads disabled."""
    from ..enrichment.ocr import EasyOCREngine

    model_dir = Path(root).resolve() / "data/models/easyocr"
    if not model_dir.is_dir():
        raise AnalysisError("ocr_models_missing", "Local OCR models are not available.",
                            "Expected the prepared EasyOCR cache in data/models/easyocr. "
                            "This action does not download models.")
    try:
        import torch

        torch.set_num_threads(4)
        engine = EasyOCREngine(languages=list(languages), gpu=False,
                               model_storage_directory=str(model_dir), download_enabled=False)
        engine.initialize()
        return engine
    except Exception as exc:
        raise AnalysisError("ocr_unavailable", "OCR could not initialize from local models.",
                            "Check the installed OCR dependencies and the language files in "
                            f"data/models/easyocr. Failure type: {type(exc).__name__}.") from exc


def _local_encoder(root: Path, identifier: str) -> Path:
    """Resolve a local directory or an already cached HF snapshot; never fetch."""
    requested = Path(identifier)
    candidate = requested if requested.is_absolute() else root / requested
    if candidate.is_dir():
        directory = candidate.resolve()
    elif requested.is_absolute() or identifier.replace("\\", "/").startswith(("data/", "./")):
        raise AnalysisError("classifier_assets_missing", "Local classifier assets are missing.",
                            "Prepare the matching encoder/tokenizer files before using this mode.")
    else:
        try:
            from huggingface_hub import snapshot_download

            directory = Path(snapshot_download(identifier, local_files_only=True))
        except Exception as exc:
            raise AnalysisError("classifier_assets_missing", "Classifier assets are not cached locally.",
                                "This mode only reads existing model snapshots and does not "
                                f"download them. Failure type: {type(exc).__name__}.") from exc
    if not (directory / "config.json").is_file():
        raise AnalysisError("classifier_assets_missing", "A local encoder configuration is missing.",
                            "The matching model snapshot must contain config.json.")
    return directory


def load_classifier(root: Path, mode: str) -> ClassifierComponents:
    """Load only an explicit compatible research checkpoint or opt-in pilot."""
    root = Path(root).resolve()
    try:
        runtime = mode_runtime(root, mode)
    except ValueError as exc:
        raise AnalysisError("invalid_mode", str(exc)) from exc
    if mode == "ocr":
        raise AnalysisError("no_classifier_in_ocr_mode", "OCR mode does not use a classifier.")
    checkpoint_path, config_path = Path(runtime["checkpoint_path"]), Path(runtime["config_path"])
    if not checkpoint_path.is_file():
        raise AnalysisError("checkpoint_missing", "The selected model checkpoint is not available.",
                            f"Expected {checkpoint_path.relative_to(root).as_posix()}. "
                            "OCR-only analysis is available independently.")
    try:
        config = load_model_config(config_path)
        config_purpose = config.raw.get("purpose")
        if mode == "pilot" and config_purpose != "internal_smoke_only":
            raise ValueError("The pilot configuration must declare internal_smoke_only.")
        if mode == "research" and config_purpose == "internal_smoke_only":
            raise ValueError("An internal smoke configuration cannot be used as a research model.")
        import torch

        from ..models.artifacts import validate_frontend_checkpoint
        from ..models.arms import build_arm

        torch.set_num_threads(4)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        threshold = validate_frontend_checkpoint(checkpoint, config)
        saved_config = checkpoint["config"]
        saved_purpose = (saved_config.get("source_config") or {}).get("purpose")
        if mode == "pilot" and saved_purpose != "internal_smoke_only":
            raise ValueError("The pilot checkpoint must declare internal_smoke_only provenance.")
        if mode == "research" and saved_purpose == "internal_smoke_only":
            raise ValueError("Internal smoke checkpoints cannot be used as research models.")
        # Compare the original saved identifiers first; localize only construction.
        text_dir = _local_encoder(root, config.student.text_encoder)
        image_dir = _local_encoder(root, config.student.image_encoder)
        architecture = replace(config.student, text_encoder=str(text_dir), image_encoder=str(image_dir))
        model = build_arm("fused", architecture, from_scratch=True)
        model.load_state_dict(checkpoint["state_dict"])
        model.to("cpu")
        model.eval()
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(str(text_dir), local_files_only=True)

        def transform(image):
            import numpy as np

            size = config.measurement.image_size
            resized = image.resize((size, size), Image.Resampling.BILINEAR)
            pixels = torch.from_numpy(np.asarray(resized, dtype=np.float32).copy()).permute(2, 0, 1) / 255
            mean = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
            std = torch.tensor((0.229, 0.224, 0.225))[:, None, None]
            return (pixels - mean) / std

        return ClassifierComponents(model=model, tokenizer=tokenizer, transform=transform,
                                    config=config, threshold=threshold, mode=mode,
                                    purpose=MODE_PURPOSES[mode], checkpoint_path=checkpoint_path,
                                    experiment_id=checkpoint.get("experiment_id"))
    except AnalysisError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise AnalysisError("checkpoint_incompatible", "The checkpoint does not match this analysis mode.",
                            str(exc)[:500]) from exc
    except Exception as exc:
        raise AnalysisError("classifier_unavailable", "The local classifier could not initialize.",
                            "Check the checkpoint, installed dependencies and cached model assets. "
                            f"Failure type: {type(exc).__name__}.") from exc


_CUES = (
    ("certainty", "Certainty or guarantee language",
     r"\b(?:guarantee(?:d)?|risk[ -]?free|zero risk|sure[ -]?shot|100\s*%)\b|गारंट[ीि]|निश्चित"),
    ("returns", "Return or payout wording",
     r"\b(?:profit|returns?|earn(?:ing|ings)?|income|payout|per day|monthly)\b|मुनाफ[ा़]|कमाई|मासिक"),
    ("amounts", "Amounts or percentages mentioned",
     r"(?:₹\s*[\d,]+(?:\.\d+)?|\b(?:rs\.?|inr)\s*[\d,]+|\d+(?:\.\d+)?\s*%)"),
    ("contact", "Contact or signup language",
     r"\b(?:telegram|whatsapp|join now|register|sign[ -]?up|dm for link)\b|रजिस्टर|जुड़ें"),
    ("authority", "Authority or public figure mentioned",
     r"\b(?:sebi|rbi|government(?: of india)?|modi|nirmala|adani|amb(?:a)?ni|murth?y)\b|सरकार|सेबी"),
    ("risk_notice", "Risk disclosure wording",
     r"\b(?:subject to market risks?|market risk|risk disclosure|past performance)\b|बाजार जोखिम"),
)


def extract_indicators(text: str) -> list[dict]:
    """Describe text features without interpreting them as fraud or legitimacy."""
    indicators = []
    for cue_id, label, pattern in _CUES:
        matches = list(dict.fromkeys(match.group(0) for match in re.finditer(pattern, text, re.I)))[:5]
        if matches:
            indicators.append({"id": cue_id, "label": label, "matches": matches,
                               "explanation": "A text cue for review; this alone does not establish "
                                              "fraud, legitimacy or an authentic endorsement."})
    return indicators


def _finite(value, default=None):
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (ValueError, TypeError):
        return default


def analyze_image(image: Image.Image, caption: str = "", mode: str = "ocr", root: Path = Path("."),
                  *, ocr_loader: Callable | None = None, model_loader: Callable | None = None) -> dict:
    """Return OCR and optional model output; operational errors remain JSON-safe.

    Pilot class scores are explicitly unvalidated. Text cues never generate a
    class label, and an OCR failure prevents any classifier invocation.
    """
    started = time.perf_counter()
    durations = {key: 0.0 for key in ("ocr_load", "ocr", "classifier_load", "inference", "total")}
    result = {"status": "error", "mode": mode, "model_status": "not_loaded",
              "purpose": MODE_PURPOSES.get(mode, "unknown"), "raw_ocr": "", "ocr": None,
              "indicators": [], "class_scores": None, "threshold": None, "label": None,
              "durations_ms": durations, "input_fingerprint": "", "warnings": [],
              "error": None, "input": None, "input_pipeline_version": INPUT_PIPELINE_VERSION}
    stage = "input"
    try:
        if mode not in MODES or not isinstance(caption, str):
            raise AnalysisError("invalid_input", "Use a supported analysis mode and a text caption.")
        normalized = canonical_image(image)
        result["input_fingerprint"] = input_fingerprint(normalized, caption, mode)
        result["input"] = {"width": normalized.width, "height": normalized.height,
                           "caption_chars": len(caption)}
        stage = "ocr_load"
        tick = time.perf_counter()
        ocr = ocr_loader() if ocr_loader is not None else load_ocr(root)
        durations["ocr_load"] = (time.perf_counter() - tick) * 1000
        stage = "ocr"
        tick = time.perf_counter()
        with tempfile.TemporaryDirectory(prefix="fipd-analysis-") as temp:
            image_path = Path(temp) / "creative.png"
            normalized.save(image_path, format="PNG")
            ocr_result = ocr.read(image_path)
        durations["ocr"] = (time.perf_counter() - tick) * 1000
        if getattr(ocr_result, "error", ""):
            raise AnalysisError("ocr_failed", "OCR could not read this image.",
                                "Try a clearer image and check the prepared local OCR language models.")
        raw_ocr = getattr(ocr_result, "text", "") or ""
        if not isinstance(raw_ocr, str):
            raise AnalysisError("ocr_failed", "OCR returned an invalid text result.")
        result["raw_ocr"] = raw_ocr.strip()
        result["ocr"] = {"backend": str(getattr(ocr_result, "backend", "")),
                         "lines": [str(line) for line in getattr(ocr_result, "lines", [])],
                         "mean_confidence": _finite(getattr(ocr_result, "mean_confidence", None)),
                         "elapsed_ms": _finite(getattr(ocr_result, "elapsed_ms", None)),
                         "script_profile": {key: _finite(value) for key, value in
                                            getattr(ocr_result, "script_profile", {}).items()}}
        result["indicators"] = extract_indicators(result["raw_ocr"] + "\n" + caption)
        if not result["raw_ocr"]:
            result["warnings"].append("No text was recognized in the image.")
        if mode == "ocr":
            result["status"] = "complete"
            return result

        stage = "classifier_load"
        tick = time.perf_counter()
        components = model_loader() if model_loader is not None else load_classifier(root, mode)
        durations["classifier_load"] = (time.perf_counter() - tick) * 1000
        if components.mode != mode or components.purpose != MODE_PURPOSES[mode]:
            raise AnalysisError("checkpoint_incompatible", "The loaded checkpoint belongs to another mode.")
        threshold = _finite(components.threshold)
        if threshold is None or not 0 <= threshold <= 1:
            raise AnalysisError("checkpoint_incompatible", "The model decision threshold is invalid.")
        stage = "inference"
        tick = time.perf_counter()
        import torch

        record = FactCheckRecord(source="frontend", source_id="upload", url="", claim_text=caption,
                                 images=[ImageAsset(url="", ocr_text=result["raw_ocr"])])
        text = assemble_text(record, components.config.text)
        encoded = components.tokenizer(text, truncation=True,
                                       max_length=components.config.measurement.max_text_tokens,
                                       padding="max_length", return_tensors="pt")
        pixels = components.transform(normalized).unsqueeze(0)
        with torch.inference_mode():
            logits = components.model(input_ids=encoded["input_ids"],
                                       attention_mask=encoded["attention_mask"], pixel_values=pixels)
            if tuple(logits.shape) != (1, 2) or not bool(torch.isfinite(logits).all()):
                raise AnalysisError("invalid_model_output", "The classifier returned invalid scores.")
            probabilities = torch.softmax(logits.float(), dim=-1)[0]
            scores = {"real": float(probabilities[0]), "fake": float(probabilities[1])}
        if (any(not math.isfinite(value) or not 0 <= value <= 1 for value in scores.values())
                or not math.isclose(sum(scores.values()), 1.0, abs_tol=1e-5)):
            raise AnalysisError("invalid_model_output", "The classifier returned invalid scores.")
        durations["inference"] = (time.perf_counter() - tick) * 1000
        result.update(status="complete", class_scores=scores, threshold=threshold,
                      label="fake" if scores["fake"] >= threshold else "real",
                      model_status="unvalidated_internal_pilot" if mode == "pilot"
                      else "compatible_checkpoint")
        if mode == "pilot":
            result["warnings"].append("Internal pilot output is unvalidated and is only a pipeline demonstration.")
        return result
    except AnalysisError as exc:
        result["error"] = {"code": exc.code, "message": exc.message, "details": exc.details}
    except Exception as exc:
        messages = {"input": "The selected image could not be prepared.",
                    "ocr_load": "OCR could not initialize from local models.",
                    "ocr": "OCR could not read this image.",
                    "classifier_load": "The selected classifier could not initialize.",
                    "inference": "The classifier could not analyze this image."}
        result["error"] = {"code": f"{stage}_failed", "message": messages[stage],
                           "details": f"Failure type: {type(exc).__name__}. Check the local assets "
                                      "and dependencies for this mode."}
    finally:
        durations["total"] = (time.perf_counter() - started) * 1000
        for key, value in durations.items():
            durations[key] = round(value, 3)
    if mode != "ocr":
        result["model_status"] = "unavailable"
    return result
