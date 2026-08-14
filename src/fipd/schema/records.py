"""Unified record schema for collected fact-check items.

Every source scraper normalises into `FactCheckRecord` so that downstream
filtering, OCR and model code never has to care where an item came from.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Label space
# ---------------------------------------------------------------------------

# Binary target used by the classifier.
LABEL_FAKE = "fake"
LABEL_REAL = "real"
LABEL_UNKNOWN = "unknown"

VALID_LABELS = {LABEL_FAKE, LABEL_REAL, LABEL_UNKNOWN}


@dataclass
class ImageAsset:
    """One image belonging to a fact-check article.

    `role` distinguishes the *scam creative itself* (what we want to train on)
    from the fact-checker's own annotations and stock art (what we must drop).
    """

    url: str
    local_path: str | None = None
    role: str = "unknown"          # creative | annotated | site_chrome | unknown
    width: int | None = None
    height: int | None = None
    sha256: str | None = None
    alt_text: str = ""
    caption: str = ""
    ocr_text: str | None = None    # filled in later by the OCR stage

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactCheckRecord:
    """A single labelled item harvested from a fact-checking archive."""

    # --- provenance -------------------------------------------------------
    source: str                     # boomlive | newschecker | altnews | factly | ...
    source_id: str                  # id within that source (WP post id, slug, ...)
    url: str

    # --- content ----------------------------------------------------------
    title: str = ""
    published_at: str | None = None  # ISO-8601
    author: str = ""
    body_text: str = ""              # article text, tags stripped
    claim_text: str = ""             # the viral claim, when the source states it
    verdict_raw: str = ""            # source's own wording: "False", "Misleading", ...

    # --- derived ----------------------------------------------------------
    label: str = LABEL_UNKNOWN
    is_finance: bool = False
    finance_score: float = 0.0
    finance_hits: list[str] = field(default_factory=list)

    # --- media ------------------------------------------------------------
    images: list[ImageAsset] = field(default_factory=list)

    # --- misc -------------------------------------------------------------
    categories: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    language: str = "en"
    collected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    raw: dict[str, Any] = field(default_factory=dict)  # untouched source payload

    # ------------------------------------------------------------------
    @property
    def uid(self) -> str:
        """Stable primary key across re-runs."""
        return hashlib.sha1(f"{self.source}:{self.source_id}".encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["uid"] = self.uid
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "FactCheckRecord":
        d = dict(d)
        d.pop("uid", None)
        imgs = [ImageAsset(**i) for i in d.pop("images", [])]
        return cls(images=imgs, **d)
