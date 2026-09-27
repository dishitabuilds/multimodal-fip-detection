"""OCR stage — extract the text that lives inside the creative.

This is the core of the project's premise. The deceptive claim sits in the
image ("40% GUARANTEED MONTHLY RETURNS", a forged SEBI registration number, a
doctored P&L figure) while the caption stays vague. Everything the text encoder
sees about that claim comes through here, so OCR quality is an upper bound on
what the text branch can contribute.

Backend strategy
----------------
Three engines behind one interface, because none of them is reliably best on
Indian scam creatives:

  * PaddleOCR  — strongest on Devanagari, heaviest install
  * EasyOCR    — good multilingual coverage, simpler install, slower per image
  * Tesseract  — weakest here, but tiny and useful as a fallback / sanity check

`OCREngine` is the abstract interface. Backends are imported lazily inside
their own `load()` so that importing this module never drags in torch, and so
a missing engine degrades to a clear error rather than an ImportError at
startup. Which backend produced a given result is recorded on the result, so
a mixed-engine dataset stays auditable.

Code-mixing
-----------
These creatives mix Devanagari Hindi, romanised Hindi and English, often in one
image. Engines are configured multilingual where they support it. The
`script_profile` on each result reports the proportion of Devanagari, Latin and
digit characters found — useful for the report, and a cheap quality signal
(a result that is 90% punctuation is almost certainly garbage).
"""

from __future__ import annotations

import json
import logging
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable, Sequence

log = logging.getLogger(__name__)

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")
_DIGIT = re.compile(r"[0-9]")


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class OCRResult:
    text: str = ""
    lines: list[str] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)
    backend: str = ""
    elapsed_ms: float = 0.0
    error: str = ""

    @property
    def mean_confidence(self) -> float:
        return sum(self.confidences) / len(self.confidences) if self.confidences else 0.0

    @property
    def script_profile(self) -> dict[str, float]:
        """Proportion of Devanagari / Latin / digit characters."""
        t = self.text or ""
        total = len(_DEVANAGARI.findall(t)) + len(_LATIN.findall(t)) + len(_DIGIT.findall(t))
        if not total:
            return {"devanagari": 0.0, "latin": 0.0, "digit": 0.0}
        return {
            "devanagari": round(len(_DEVANAGARI.findall(t)) / total, 3),
            "latin": round(len(_LATIN.findall(t)) / total, 3),
            "digit": round(len(_DIGIT.findall(t)) / total, 3),
        }

    def looks_like_garbage(self, min_chars: int = 8, min_alnum_ratio: float = 0.5) -> bool:
        """Cheap quality gate before this text reaches the encoder."""
        t = (self.text or "").strip()
        if len(t) < min_chars:
            return True
        alnum = sum(c.isalnum() for c in t)
        return (alnum / len(t)) < min_alnum_ratio

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["mean_confidence"] = round(self.mean_confidence, 4)
        d["script_profile"] = self.script_profile
        return d


# ---------------------------------------------------------------------------
# Backend interface
# ---------------------------------------------------------------------------

class OCREngine(ABC):
    name: str = "base"

    def __init__(self, languages: Sequence[str] | None = None, **kw: Any) -> None:
        self.languages = list(languages or ["en", "hi"])
        self.options = kw
        self._loaded = False

    @abstractmethod
    def load(self) -> None:
        """Import and initialise the underlying engine. Called lazily."""

    @abstractmethod
    def _run(self, image_path: str) -> tuple[list[str], list[float]]:
        """Return (lines, confidences)."""

    def read(self, image_path: str | Path) -> OCRResult:
        image_path = str(image_path)
        if not self._loaded:
            try:
                self.load()
                self._loaded = True
            except Exception as e:
                return OCRResult(backend=self.name, error=f"load failed: {e}")

        t0 = time.perf_counter()
        try:
            lines, confs = self._run(image_path)
        except Exception as e:
            return OCRResult(backend=self.name, error=str(e),
                             elapsed_ms=(time.perf_counter() - t0) * 1000)

        return OCRResult(
            text="\n".join(lines).strip(),
            lines=lines,
            confidences=confs,
            backend=self.name,
            elapsed_ms=round((time.perf_counter() - t0) * 1000, 1),
        )


class PaddleOCREngine(OCREngine):
    """Best Devanagari accuracy in our experience; heaviest dependency."""

    name = "paddleocr"

    def load(self) -> None:
        from paddleocr import PaddleOCR  # noqa: PLC0415

        # PaddleOCR takes ONE language per model. 'devanagari' covers Hindi and
        # also recognises Latin characters, which suits code-mixed creatives
        # better than running the English model over Hindi text.
        lang = "devanagari" if "hi" in self.languages else "en"
        self._engine = PaddleOCR(
            use_angle_cls=True, lang=lang, show_log=False, **self.options
        )

    def _run(self, image_path: str) -> tuple[list[str], list[float]]:
        raw = self._engine.ocr(image_path, cls=True)
        lines: list[str] = []
        confs: list[float] = []
        for page in raw or []:
            for entry in page or []:
                try:
                    text, conf = entry[1][0], float(entry[1][1])
                except (IndexError, TypeError, ValueError):
                    continue
                lines.append(text)
                confs.append(conf)
        return lines, confs


class EasyOCREngine(OCREngine):
    """Simpler install, handles multiple languages in one pass."""

    name = "easyocr"

    def load(self) -> None:
        import easyocr  # noqa: PLC0415
        from pathlib import Path  # noqa: PLC0415

        model_dir = self.options.get("model_storage_directory")
        if not model_dir:
            local_dir = Path("data/models/easyocr")
            if local_dir.exists():
                model_dir = str(local_dir.resolve())

        self._engine = easyocr.Reader(
            self.languages,
            gpu=self.options.get("gpu", False),
            model_storage_directory=model_dir,
            verbose=False,
        )

    def _run(self, image_path: str) -> tuple[list[str], list[float]]:
        raw = self._engine.readtext(image_path, detail=1)
        lines, confs = [], []
        for entry in raw or []:
            try:
                text, conf = entry[1], float(entry[2])
            except (IndexError, TypeError, ValueError):
                continue
            lines.append(text)
            confs.append(conf)
        return lines, confs


class TesseractEngine(OCREngine):
    """Fallback. Needs the Tesseract binary plus the `hin` traineddata."""

    name = "tesseract"

    def load(self) -> None:
        import pytesseract  # noqa: PLC0415

        self._pt = pytesseract
        binary = self.options.get("tesseract_cmd")
        if binary:
            pytesseract.pytesseract.tesseract_cmd = binary
        pytesseract.get_tesseract_version()  # raises if the binary is missing

    def _run(self, image_path: str) -> tuple[list[str], list[float]]:
        langs = "+".join({"hi": "hin", "en": "eng"}.get(l, l) for l in self.languages)
        data = self._pt.image_to_data(
            image_path, lang=langs, output_type=self._pt.Output.DICT
        )
        lines, confs = [], []
        for text, conf in zip(data.get("text", []), data.get("conf", [])):
            text = (text or "").strip()
            try:
                conf = float(conf)
            except (TypeError, ValueError):
                continue
            if text and conf >= 0:
                lines.append(text)
                confs.append(conf / 100.0)
        return lines, confs


BACKENDS: dict[str, type[OCREngine]] = {
    "paddleocr": PaddleOCREngine,
    "easyocr": EasyOCREngine,
    "tesseract": TesseractEngine,
}


def get_engine(name: str, languages: Sequence[str] | None = None, **kw: Any) -> OCREngine:
    if name not in BACKENDS:
        raise ValueError(f"unknown OCR backend {name!r}; choose from {sorted(BACKENDS)}")
    return BACKENDS[name](languages=languages, **kw)


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

class OCRCache:
    """Sidecar JSON cache keyed by image path.

    OCR is by far the slowest stage in the pipeline, and re-running it because
    a later step crashed is intolerable. Results are flushed to disk as they
    are produced, so the work already done always survives.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.data: dict[str, dict] = {}
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                log.warning("corrupt OCR cache at %s, starting fresh", self.path)

    def get(self, key: str) -> OCRResult | None:
        d = self.data.get(key)
        if not d:
            return None
        return OCRResult(
            text=d.get("text", ""), lines=d.get("lines", []),
            confidences=d.get("confidences", []), backend=d.get("backend", ""),
            elapsed_ms=d.get("elapsed_ms", 0.0), error=d.get("error", ""),
        )

    def put(self, key: str, result: OCRResult) -> None:
        self.data[key] = result.to_dict()

    def flush(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=1),
                             encoding="utf-8")


def ocr_records(
    records: Iterable,
    engine: OCREngine,
    data_dir: str | Path = "data",
    cache_path: str | Path | None = None,
    flush_every: int = 25,
    skip_annotated: bool = True,
) -> dict[str, int]:
    """Run OCR over every downloaded image and fill `ImageAsset.ocr_text`.

    `skip_annotated` leaves out images tagged as fact-checker annotations —
    running OCR on a verdict stamp would inject the word "FALSE" straight into
    the text branch, which is exactly the leakage the role tagging exists to
    prevent.
    """
    data_dir = Path(data_dir)
    cache = OCRCache(cache_path or data_dir / "interim" / "ocr_cache.json")
    stats = {"images": 0, "ocr_run": 0, "cache_hits": 0,
             "skipped_annotated": 0, "missing": 0, "errors": 0, "garbage": 0}
    n = 0

    for rec in records:
        for asset in rec.images:
            if not asset.local_path:
                continue
            stats["images"] += 1

            if skip_annotated and asset.role == "annotated":
                stats["skipped_annotated"] += 1
                continue

            path = data_dir / asset.local_path
            if not path.exists():
                stats["missing"] += 1
                continue

            key = str(asset.local_path)
            result = cache.get(key)
            if result is not None:
                stats["cache_hits"] += 1
            else:
                result = engine.read(path)
                cache.put(key, result)
                stats["ocr_run"] += 1
                n += 1
                if n % flush_every == 0:
                    cache.flush()

            if result.error:
                stats["errors"] += 1
                continue
            if result.looks_like_garbage():
                stats["garbage"] += 1
                continue

            asset.ocr_text = result.text

    cache.flush()
    return stats
