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

MODES = ("ocr", "research", "pilot", "svm", "multimodal")
SVM_EXPERIMENT_ID = "visual-source-holdout-20261113"
SVM_CHECKPOINT_RELATIVE = Path("demo_bundle/models/fused_linear_svm.joblib")
SVM_WEIGHTS_RELATIVE = Path("demo_bundle/weights/mobilenet_v3_small-047dcff4.pth")
SVM_MANIFEST_RELATIVE = Path("demo_bundle/manifest.json")
SVM_EXPERIMENT_RELATIVE = Path("demo_bundle/experiment.json")
MULTIMODAL_MODEL_RELATIVE = Path("data/processed/multimodal_generalization/model.joblib")
MULTIMODAL_REPORT_RELATIVE = Path("data/processed/multimodal_generalization/report.json")
MULTIMODAL_REVIEW_LOW = 0.45
MULTIMODAL_REVIEW_HIGH = 0.55


def multimodal_decision(probability: float) -> tuple[str | None, str | None]:
    """Return a label outside the inclusive uncertainty band only."""
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("The calibrated scam probability must be between 0 and 1.")
    if MULTIMODAL_REVIEW_LOW <= probability <= MULTIMODAL_REVIEW_HIGH:
        return None, "uncertain_prediction"
    return ("fake" if probability > 0.50 else "real"), None
RESEARCH_CHECKPOINT_RELATIVE = Path("data/checkpoints/fused_seed42.pt")
PILOT_CHECKPOINT_RELATIVE = Path(
    "data/checkpoints/student/internal-ai-pilot-20261025/fused_seed42.pt")
RESEARCH_CONFIG_RELATIVE = Path("configs/model.yaml")
PILOT_CONFIG_RELATIVE = Path("configs/tiny-smoke.yaml")
MODE_PURPOSES = {"ocr": "text_extraction_only", "research": "research_inference",
                 "pilot": "internal_smoke_only", "svm": "internal_smoke_only",
                 "multimodal": "cross_format_specialist"}


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


@dataclass
class SVMClassifierComponents:
    artifact: dict
    encoder: Any
    artifact_sha256: str
    checkpoint_path: Path
    threshold: float = 0.0
    mode: str = "svm"
    purpose: str = "internal_smoke_only"


def _load_svm_classifier(root: Path) -> SVMClassifierComponents:
    """Load only the packaged presentation model and its matching image weights."""
    runtime = mode_runtime(root, "svm")
    if not runtime["checkpoint_available"]:
        raise AnalysisError("svm_assets_missing", "The classification model files are missing.",
                            "Restore demo_bundle with its models, weights, manifest.json and "
                            "experiment.json. Text extraction remains available.")
    try:
        manifest = json.loads((root / SVM_MANIFEST_RELATIVE).read_text())
        for relative in (SVM_CHECKPOINT_RELATIVE, SVM_WEIGHTS_RELATIVE, SVM_EXPERIMENT_RELATIVE):
            path = root / relative
            expected = manifest[relative.relative_to("demo_bundle").as_posix()]
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if digest != expected["sha256"] or path.stat().st_size != expected["bytes"]:
                raise ValueError(f"Packaged file has changed: {relative.name}")
        experiment = json.loads((root / SVM_EXPERIMENT_RELATIVE).read_text())
        if (experiment["experiment_id"] != SVM_EXPERIMENT_ID
                or {k: sum(v.values()) for k, v in experiment["counts"].items()}
                != {"train": 31, "val": 6, "test": 10}):
            raise ValueError("Bundle does not describe the presentation's 47-image experiment")
        # Verify bytes before joblib deserialization, then reuse its fitted preprocessing.
        from ..models.classical import load_lab_artifact, MobileNetFeatures

        artifact = load_lab_artifact(root / SVM_CHECKPOINT_RELATIVE, allow_internal=True)
        if (artifact["experiment_id"] != SVM_EXPERIMENT_ID or artifact["arm"] != "fused"
                or artifact["model"] != "linear_svm" or artifact["threshold"] != 0.0
                or artifact["purpose"] != "internal_smoke_only"
                or artifact["cohort_hash"] != experiment["cohort_hash"]
                or artifact["text_config"] != experiment["text_config"]):
            raise ValueError("Classifier does not match the selected pilot experiment")
        encoder = MobileNetFeatures(root / SVM_WEIGHTS_RELATIVE)
        if artifact["image_weights_sha256"] != encoder.weights_sha256:
            raise ValueError("Image encoder does not match the saved classifier")
        return SVMClassifierComponents(
            artifact=artifact, encoder=encoder, checkpoint_path=root / SVM_CHECKPOINT_RELATIVE,
            artifact_sha256=manifest["models/fused_linear_svm.joblib"]["sha256"])
    except (ValueError, KeyError, TypeError) as exc:
        raise AnalysisError("svm_bundle_incompatible", "The classification model could not be verified.",
                            str(exc)[:500]) from exc
    except Exception as exc:
        raise AnalysisError("svm_unavailable", "The classification model could not initialize.",
                            "Use the model's matching scikit-learn environment and prepared local "
                            f"assets. {type(exc).__name__}: {str(exc)[:250]}") from exc


@dataclass
class MultimodalClassifierComponents:
    artifact: dict
    encoder: Any
    artifact_sha256: str
    checkpoint_path: Path
    threshold: float = 0.50
    mode: str = "multimodal"
    purpose: str = "cross_format_specialist"


def _load_multimodal_classifier(root: Path) -> MultimodalClassifierComponents:
    """Load the 120-image multimodal generalization model and matching image encoder."""
    runtime = mode_runtime(root, "multimodal")
    if not runtime["checkpoint_available"]:
        raise AnalysisError("multimodal_assets_missing", "The multimodal model files are missing.",
                            "Expected model at data/processed/multimodal_generalization/model.joblib "
                            "and weights at demo_bundle/weights/mobilenet_v3_small-047dcff4.pth.")
    try:
        import joblib
        from ..models.classical import MobileNetFeatures

        model_path = root / MULTIMODAL_MODEL_RELATIVE
        weights_path = root / SVM_WEIGHTS_RELATIVE
        digest = hashlib.sha256(model_path.read_bytes()).hexdigest()

        artifact = joblib.load(model_path)
        if not isinstance(artifact, dict) or "models" not in artifact or "specialists" not in artifact["models"]:
            raise ValueError("Artifact does not contain expected format specialist models")

        encoder = MobileNetFeatures(weights_path)
        if artifact.get("image_weights_sha256") != encoder.weights_sha256:
            raise ValueError("Image encoder weights do not match the saved multimodal classifier")

        return MultimodalClassifierComponents(
            artifact=artifact, encoder=encoder, artifact_sha256=digest,
            checkpoint_path=model_path, threshold=0.50,
        )
    except AnalysisError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise AnalysisError("multimodal_bundle_incompatible",
                            "The multimodal model could not be verified.",
                            str(exc)[:500]) from exc
    except Exception as exc:
        raise AnalysisError("multimodal_unavailable",
                            "The multimodal model could not initialize.",
                            f"Failure type: {type(exc).__name__}: {str(exc)[:250]}") from exc


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
        raise ValueError("Unknown analysis mode; choose OCR, SVM, research or pilot.")
    root = Path(root).resolve()
    if mode == "svm":
        paths = [root / p for p in (SVM_CHECKPOINT_RELATIVE, SVM_WEIGHTS_RELATIVE,
                                    SVM_MANIFEST_RELATIVE, SVM_EXPERIMENT_RELATIVE)]
        available = all(p.is_file() for p in paths)
        return {"mode": mode, "purpose": MODE_PURPOSES[mode],
                "model_status": "not_loaded_bundle_present" if available else "unavailable",
                "checkpoint_path": str(paths[0]), "config_path": str(paths[2]),
                "asset_paths": [str(paths[1]), str(paths[3])],
                "checkpoint_available": available, "unvalidated": True,
                "experiment_id": SVM_EXPERIMENT_ID}
    if mode == "multimodal":
        checkpoint = root / MULTIMODAL_MODEL_RELATIVE
        weights = root / SVM_WEIGHTS_RELATIVE
        available = checkpoint.is_file() and weights.is_file()
        return {"mode": mode, "purpose": MODE_PURPOSES[mode],
                "model_status": "not_loaded_multimodal_present" if available else "unavailable",
                "checkpoint_path": str(checkpoint), "config_path": str(checkpoint),
                "asset_paths": [str(weights)],
                "checkpoint_available": available, "unvalidated": True,
                "experiment_id": "multimodal_generalization_v1"}
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
                               model_storage_directory=str(model_dir),
                               download_enabled=False)
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


def load_classifier(root: Path, mode: str) -> ClassifierComponents | SVMClassifierComponents | MultimodalClassifierComponents:
    """Load only an explicit compatible research checkpoint, SVM, or multimodal model."""
    root = Path(root).resolve()
    if mode == "svm":
        return _load_svm_classifier(root)
    if mode == "multimodal":
        return _load_multimodal_classifier(root)
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


def phishing_out_of_scope(text: str) -> bool:
    """Abstain on reward lures outside the investment-promotion training domain."""
    reward = re.search(r"\b(?:win|won|winner|prize|reward|gift\s*card|congratulations)\b", text, re.I)
    action = re.search(r"\b(?:claim|click|scan|(?:qr|or)\s*cod[ec]|review\s+(?:the\s+)?policy)\b", text, re.I)
    investment = re.search(r"\b(?:invest|investment|mutual\s+fund|shares?|stocks?|portfolio|returns?)\b", text, re.I)
    return bool(reward and action and not investment)


def decode_qr(image: Image.Image) -> list[str]:
    """Decode local QR content without opening URLs or making network requests."""
    try:
        import cv2
        import numpy as np

        pixels = cv2.cvtColor(np.asarray(canonical_image(image)), cv2.COLOR_RGB2BGR)
        payload, _, _ = cv2.QRCodeDetector().detectAndDecode(pixels)
        return [payload[:2048]] if payload else []
    except Exception:  # QR decoding is optional evidence, not a classification dependency.
        return []


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
              "decision_score": None, "score_kind": None, "experiment_id": None,
              "review_reason": None, "model_label": None, "qr_payloads": [],
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
        result["qr_payloads"] = decode_qr(normalized)
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
        if mode == "svm":
            from ..models.classical import predict_lab

            if not result["raw_ocr"] and not caption.strip():
                raise AnalysisError("classification_text_missing",
                                    "No readable text was found for image-and-text classification.",
                                    "Use a clearer image or supply its original caption.")
            prediction = predict_lab(components.artifact, result["raw_ocr"], caption,
                                     image=normalized, encoder=components.encoder)
            score = _finite(prediction["score"])
            if (score is None or prediction["score_kind"] != "decision_margin"
                    or prediction["label"] not in {"fake", "real"}
                    or prediction["threshold"] != 0.0
                    or prediction["experiment_id"] != SVM_EXPERIMENT_ID):
                raise AnalysisError("invalid_model_output", "The classifier returned invalid output.")
            durations["inference"] = (time.perf_counter() - tick) * 1000
            result.update(status="complete", label=prediction["label"], decision_score=score,
                          score_kind="decision_margin", threshold=0.0,
                          experiment_id=prediction["experiment_id"],
                          model_status="preliminary_47_image_svm",
                          artifact_sha256=components.artifact_sha256)
            from ..models.multimodal_evidence import (
                detect_format, extract_evidence_indicators, normalize_multimodal_text
            )
            full_text = (result["raw_ocr"] + "\n" + caption).strip()
            w, h = normalized.size
            detected_fmt = detect_format(w, h, full_text, result["qr_payloads"])
            evidence = extract_evidence_indicators(full_text, result["qr_payloads"])
            result["detected_format"] = detected_fmt

            # Format specialist evaluation if trained generalization artifact exists
            gen_path = root / "data/processed/multimodal_generalization/model.joblib"
            if gen_path.is_file() and hasattr(components.encoder, "encode_images"):
                try:
                    import joblib
                    gen_model = joblib.load(gen_path)
                    v_feat = components.encoder.encode_images([normalized])
                    norm_txt = normalize_multimodal_text(full_text)
                    specialists = gen_model["models"]["specialists"]
                    sp = specialists.get(detected_fmt, specialists["investment_graphic"])
                    X_sp = sp["space"].transform([norm_txt], images=v_feat)
                    sp_margin = float(sp["classifier"].decision_function(X_sp)[0])
                    sp_prob = float(sp["calibrator"].predict_proba([sp_margin])[0, 1])
                    result["multimodal_assessment"] = {
                        "model_version": gen_model["model_version"],
                        "detected_format": detected_fmt,
                        "specialist_margin": round(sp_margin, 4),
                        "calibrated_probability": round(sp_prob, 4),
                        "indicators": evidence["indicators"],
                        "visible_urls": evidence["visible_urls"],
                        "domains": evidence["domains"],
                        "is_simulation": evidence["is_simulation"],
                    }
                except Exception:
                    pass

            is_phish_lure = phishing_out_of_scope(full_text)
            is_sim = evidence.get("is_simulation", False)
            if is_phish_lure or is_sim or (detected_fmt != "investment_graphic" and detected_fmt in {"email_screenshot", "sms_screenshot"} and score < 0.0):
                reason = "phishing_out_of_scope" if is_phish_lure else ("simulated_phishing_qr" if is_sim else "format_out_of_scope")
                result.update(label=None, model_label=prediction["label"],
                              review_reason=reason)
                if is_sim:
                    result["warnings"].append(
                        "QR code payload contains simulated phishing text. "
                        "This appears to be an educational simulation rather than a live malicious link.")
                elif is_phish_lure:
                    result["warnings"].append(
                        "Reward and action wording suggests a possible phishing lure. "
                        "This investment-promotion model was not trained to assess phishing; "
                        "verify the sender and destination independently.")
                else:
                    result["warnings"].append(
                        f"Detected {detected_fmt.replace('_', ' ')} layout. "
                        "The 47-image pilot model was trained on investment promotions; review required.")
            result["warnings"].append("47-image pilot model; predictions support review, "
                                      "not investment-safety certification.")
            return result
        if mode == "multimodal":
            from ..models.multimodal_evidence import (
                detect_format, extract_evidence_indicators, normalize_multimodal_text
            )

            if not result["raw_ocr"] and not caption.strip():
                raise AnalysisError("classification_text_missing",
                                    "No readable text was found for cross-format analysis.",
                                    "Use a clearer image or supply its original caption.")
            full_text = (result["raw_ocr"] + "\n" + caption).strip()
            w, h = normalized.size
            qrs = result.get("qr_payloads", [])
            detected_fmt = detect_format(w, h, full_text, qrs)
            evidence = extract_evidence_indicators(full_text, qrs)
            result["detected_format"] = detected_fmt

            v_feat = components.encoder.encode_images([normalized])
            norm_txt = normalize_multimodal_text(full_text)

            specialists = components.artifact["models"]["specialists"]
            sp = specialists.get(detected_fmt, specialists["investment_graphic"])
            X_sp = sp["space"].transform([norm_txt], images=v_feat)
            sp_margin = float(sp["classifier"].decision_function(X_sp)[0])
            sp_prob = float(sp["calibrator"].predict_proba([sp_margin])[0, 1])

            durations["inference"] = (time.perf_counter() - tick) * 1000

            scores = {"fake": round(sp_prob, 4), "real": round(1.0 - sp_prob, 4)}
            assigned_label, review_reason = multimodal_decision(sp_prob)
            raw_label = "fake" if sp_prob >= 0.50 else "real"

            result.update(
                status="complete",
                label=assigned_label,
                model_label=raw_label,
                review_reason=review_reason,
                class_scores=scores,
                calibrated_probability=round(sp_prob, 4),
                decision_score=round(sp_margin, 4),
                score_kind="calibrated_probability",
                threshold=0.50,
                experiment_id=components.artifact.get("model_version", "multimodal_generalization_v1"),
                model_status="120_image_multimodal_specialist",
                artifact_sha256=getattr(components, "artifact_sha256", "local"),
                evidence_details=evidence,
            )

            result["warnings"].append(
                f"120-image multimodal specialist model ({detected_fmt.replace('_', ' ')}). "
                f"Decision: Real below 45%, Review required from 45% to 55%, "
                f"Fake above 55% · Calibrated scam probability: {sp_prob:.1%}."
            )
            return result
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
