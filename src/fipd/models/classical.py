"""CPU lab models: train-only TF-IDF, frozen MobileNet features and linear heads.

SVM outputs are signed margins, never probabilities. Image weights are prepared
explicitly; feature extraction and prediction never initiate a download.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from .config import TextConfig
from ..datasets.examples import INPUT_PIPELINE_VERSION, assemble_text
from ..schema.records import FactCheckRecord, ImageAsset

LAB_PIPELINE_VERSION = "tfidf-mobilenet-linear-v1"
DEFAULT_IMAGE_WEIGHTS = Path("data/models/mobilenet_v3_small/mobilenet_v3_small-047dcff4.pth")
MODEL_NAMES = ("linear_svm", "logistic_regression", "naive_bayes")


def canonical_image(image):
    """Use the same orientation/white alpha background in training and uploads."""
    from PIL import Image, ImageOps

    oriented = ImageOps.exif_transpose(image)
    if "A" in oriented.getbands() or "transparency" in oriented.info:
        rgba = oriented.convert("RGBA")
        return Image.alpha_composite(Image.new("RGBA", rgba.size, "white"), rgba).convert("RGB")
    return oriented.convert("RGB")


def prepare_image_weights(path: str | Path = DEFAULT_IMAGE_WEIGHTS) -> Path:
    """Explicit one-time download from torchvision's official, hash-checked URL."""
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights

    path = Path(path)
    if path.is_file():
        _check_weights(path)
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".download")
    try:
        torch.hub.download_url_to_file(MobileNet_V3_Small_Weights.IMAGENET1K_V1.url,
                                      str(temporary), hash_prefix="047dcff4")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _check_weights(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError("MobileNet weights are missing. Run scripts/10_train_lab.py "
                                "--prepare-image-model once, or use --arms text_only.")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if not digest.startswith("047dcff4"):
        raise ValueError("MobileNet weights do not match torchvision's pretrained checkpoint")
    return digest


class MobileNetFeatures:
    """Frozen ImageNet encoder; outputs 576 features per image on CPU."""

    def __init__(self, weights_path=DEFAULT_IMAGE_WEIGHTS, batch_size=16, threads=4):
        import torch
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        if batch_size < 1 or threads < 1:
            raise ValueError("batch size and CPU threads must be positive")
        self.weights_sha256 = _check_weights(Path(weights_path))
        self.batch_size = batch_size
        torch.set_num_threads(threads)
        self.model = mobilenet_v3_small(weights=None)
        self.model.load_state_dict(torch.load(weights_path, map_location="cpu", weights_only=True))
        self.model.classifier = torch.nn.Identity()
        self.model.requires_grad_(False)
        self.model.eval()
        self.transform = MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()

    def encode_images(self, images) -> np.ndarray:
        import torch

        tensors = [self.transform(canonical_image(image)) for image in images]
        if not tensors:
            return np.empty((0, 576), dtype=np.float32)
        with torch.inference_mode():
            features = self.model(torch.stack(tensors)).numpy().astype(np.float32)
        if features.shape != (len(tensors), 576) or not np.isfinite(features).all():
            raise ValueError("MobileNet returned invalid image features")
        return features

    def encode_paths(self, paths) -> np.ndarray:
        from PIL import Image

        paths = list(paths)
        batches = []
        for start in range(0, len(paths), self.batch_size):
            images = []
            for path in paths[start:start + self.batch_size]:
                with Image.open(path) as image:
                    images.append(canonical_image(image))
            batches.append(self.encode_images(images))
        return np.vstack(batches) if batches else np.empty((0, 576), dtype=np.float32)


@dataclass
class LabFeatureSpace:
    """Fitted preprocessing saved with each classifier for identical inference."""

    arm: str = "text_only"
    max_features: int = 10000  # per word/character branch
    vectorizer: Any = None
    image_scaler: Any = None

    def fit(self, texts, images=None):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import StandardScaler

        if self.arm not in {"text_only", "image_only", "fused"}:
            raise ValueError(f"unknown feature arm: {self.arm}")
        if self.max_features < 1:
            raise ValueError("max_features must be positive")
        if self.arm != "image_only":
            # Character features retain Hindi combining marks and tolerate OCR errors.
            self.vectorizer = FeatureUnion([
                ("word", TfidfVectorizer(ngram_range=(1, 2), token_pattern=r"(?u)\b\w+\b",
                                         max_features=self.max_features, sublinear_tf=True,
                                         dtype=np.float32)),
                ("character", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                                              max_features=self.max_features,
                                              sublinear_tf=True, dtype=np.float32)),
            ])
            self.vectorizer.fit(texts)
        if self.arm != "text_only":
            values = self._images(images, len(texts))
            self.image_scaler = StandardScaler().fit(values)
        return self

    @staticmethod
    def _images(images, count):
        values = np.asarray(images, dtype=np.float32)
        if values.ndim != 2 or values.shape[0] != count or not np.isfinite(values).all():
            raise ValueError("finite image features must match the number of text rows")
        return values

    def transform(self, texts, images=None):
        from scipy.sparse import csr_matrix, hstack
        from sklearn.preprocessing import normalize

        blocks = []
        if self.arm != "image_only":
            blocks.append(normalize(self.vectorizer.transform(texts)))
        if self.arm != "text_only":
            values = self._images(images, len(texts))
            blocks.append(csr_matrix(normalize(self.image_scaler.transform(values))))
        features = normalize(hstack(blocks, format="csr"))
        if not np.isfinite(features.data).all():
            raise ValueError("preprocessing returned nonfinite values")
        return features


def build_classifier(name, value=1.0, seed=42):
    if name == "linear_svm":
        from sklearn.svm import LinearSVC

        return LinearSVC(C=value, class_weight="balanced", dual="auto",
                         max_iter=10000, random_state=seed)
    if name == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(C=value, class_weight="balanced", solver="liblinear",
                                  max_iter=2000, random_state=seed)
    if name == "naive_bayes":
        from sklearn.naive_bayes import MultinomialNB

        return MultinomialNB(alpha=value)
    raise ValueError(f"unknown classifier: {name}")


def classifier_scores(classifier, features):
    """Return fake-class scores and their native threshold/meaning."""
    if list(classifier.classes_) != [0, 1]:
        raise ValueError("classifier must contain real=0 and fake=1 in that order")
    if hasattr(classifier, "decision_function"):
        scores = classifier.decision_function(features)
        # Logistic regression has a margin too; preserve its useful probability output.
        if not hasattr(classifier, "predict_proba"):
            return np.asarray(scores), 0.0, "decision_margin"
    scores = classifier.predict_proba(features)[:, 1]
    return np.asarray(scores), 0.5, "uncalibrated_class_probability"


def load_lab_artifact(path, *, allow_internal=False):
    """Load an artifact generated locally by this project (joblib uses pickle)."""
    import joblib
    import sklearn

    artifact = joblib.load(path)
    if (not isinstance(artifact, dict) or artifact.get("schema_version") != 1
            or artifact.get("pipeline_version") != LAB_PIPELINE_VERSION
            or artifact.get("input_pipeline_version") != INPUT_PIPELINE_VERSION
            or artifact.get("label_mapping") != {"real": 0, "fake": 1}):
        raise ValueError("incompatible lab model artifact")
    if artifact.get("sklearn_version") != sklearn.__version__:
        raise ValueError("artifact requires the scikit-learn version used to train it")
    if artifact.get("purpose") not in {"research", "internal_smoke_only"}:
        raise ValueError("artifact has an invalid purpose")
    if artifact["purpose"] == "internal_smoke_only" and not allow_internal:
        raise ValueError("internal pilot artifact requires explicit --internal-pilot")
    if (not isinstance(artifact.get("features"), LabFeatureSpace)
            or artifact["features"].arm != artifact.get("arm")
            or artifact.get("classifier") is None
            or not np.isfinite(artifact.get("threshold", np.nan))):
        raise ValueError("artifact preprocessing/classifier is incomplete")
    return artifact


def predict_lab(artifact, ocr_text="", caption="", image=None, encoder=None):
    cfg = TextConfig(**artifact["text_config"])
    record = FactCheckRecord(source="lab", source_id="input", url="", claim_text=caption,
                             images=[ImageAsset(url="", ocr_text=ocr_text)])
    text = assemble_text(record, cfg)
    if artifact["arm"] != "image_only" and not text.strip():
        raise ValueError("provide OCR text or an original caption for this model")
    images = None
    if artifact["arm"] != "text_only":
        if image is None or encoder is None:
            raise ValueError("this model requires an image and the matching MobileNet encoder")
        if encoder.weights_sha256 != artifact["image_weights_sha256"]:
            raise ValueError("image encoder weights differ from the training run")
        images = encoder.encode_images([image])
    features = artifact["features"].transform([text], images)
    scores, _, kind = classifier_scores(artifact["classifier"], features)
    score = float(scores[0])
    if not np.isfinite(score):
        raise ValueError("classifier returned a nonfinite score")
    return {"label": "fake" if score >= artifact["threshold"] else "real",
            "score": score, "score_kind": kind, "threshold": artifact["threshold"],
            "model": artifact["model"], "arm": artifact["arm"],
            "purpose": artifact["purpose"], "experiment_id": artifact["experiment_id"],
            "unvalidated": artifact["purpose"] == "internal_smoke_only"}
