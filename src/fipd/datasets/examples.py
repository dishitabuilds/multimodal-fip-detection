"""Curated records -> flat training examples, with the split already applied.

This is the seam between the data pipeline and the model. Everything above it
deals in `FactCheckRecord`s, which carry provenance, raw payloads and a list of
images; everything below wants one row per trainable item with a label, a
string and an image path. Doing that conversion in one place means the three
ablation arms are guaranteed to see the same examples, which is the only way
their numbers are comparable.

No torch here on purpose — the split, the label mapping and the text assembly
are all testable without it, and they are where the mistakes that quietly
invalidate results actually live.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Sequence

from ..enrichment.translit import build_text_input
from ..models.config import TextConfig
from ..schema.records import LABEL_FAKE, LABEL_REAL, FactCheckRecord
from ..utils.io import read_records

log = logging.getLogger(__name__)

#: The model's target space. `unknown` never becomes an example — an unlabelled
#: item is not a negative, and treating it as one poisons the negative class.
LABEL_TO_ID = {LABEL_REAL: 0, LABEL_FAKE: 1}
ID_TO_LABEL = {v: k for k, v in LABEL_TO_ID.items()}


@dataclass
class Example:
    """One trainable item: a piece of text, an image, a label."""

    uid: str
    text: str
    label: int
    image_path: str | None = None
    source: str = ""
    cluster_id: str = ""
    split: str = ""
    #: True when the caption alone carries no fraud signal and the claim is
    #: image-only. The Gate 4 evaluation subset is built from these.
    image_only_claim: bool = False
    meta: dict = field(default_factory=dict)

    @property
    def has_image(self) -> bool:
        return bool(self.image_path)


# ---------------------------------------------------------------------------


def pick_image(record: FactCheckRecord) -> str | None:
    """Choose the one image to train on: the scam creative, never an annotation.

    A fact-checker's own graphic has the verdict stamped on it. An image encoder
    handed those learns to read the stamp, scores beautifully, and has learned
    nothing about fraud — the `role` field exists to prevent exactly that, so
    annotated assets are excluded here rather than filtered downstream.
    """
    creatives = [a for a in record.images
                 if a.local_path and a.role not in {"annotated", "site_chrome"}]
    if not creatives:
        return None
    # Largest declared area first; the biggest asset is the creative itself
    # rather than a thumbnail, and OCR needs the pixels.
    creatives.sort(key=lambda a: -((a.width or 0) * (a.height or 0)))
    return creatives[0].local_path


def assemble_text(record: FactCheckRecord, cfg: TextConfig | None = None) -> str:
    """Build the text-branch input for one record, per configs/model.yaml."""
    cfg = cfg or TextConfig()
    ocr = " ".join(a.ocr_text for a in record.images if a.ocr_text) if \
        "ocr_text" in cfg.fields else ""
    claim = record.claim_text if "claim_text" in cfg.fields else ""
    title = record.title if "title" in cfg.fields else ""

    text = build_text_input(
        caption=claim,
        ocr_text=ocr,
        title=title,
        separator=cfg.separator,
        normalise=cfg.transliterate,
        max_chars=cfg.max_chars,
    )
    return text.lower() if cfg.lowercase else text


def is_image_only_claim(record: FactCheckRecord, threshold: float = 4.0) -> bool:
    """Whether the deception is carried by the image and not the caption.

    Defined as: the in-image text clears the finance filter and the caption
    side, scored on its own, does not. These are the posts where a text-only
    classifier has nothing to go on, so they are the Gate 4 subset — the single
    most convincing experiment in the project, because fusion beating text-only
    *here* is the thesis, and overall F1 is not.

    KNOWN LIMITATION, and it belongs in the limitations section of the report:
    fact-check archives do not publish the original post's caption. The closest
    field is `claim_text`, the fact-checker's rendering of the viral claim, and
    that is what the caption side is scored on here. `title` is deliberately
    NOT used — it is the fact-checker's own headline, which almost always names
    the scam ("Viral post falsely claims SEBI approved this trading app"), so
    scoring it would make the caption side look informative for every record and
    leave this subset permanently empty.

    So this is a proxy for a benign caption, not a measurement of one. Build the
    subset with it, then hand-check the members before any claim rests on them.
    """
    from ..curation.finance_filter import score_text

    ocr = " ".join(a.ocr_text for a in record.images if a.ocr_text)
    if not ocr.strip():
        return False

    caption_side = score_text(claim=record.claim_text, threshold=threshold)
    image_side = score_text(image_text=ocr, threshold=threshold)
    return image_side.is_finance and not caption_side.is_finance


# ---------------------------------------------------------------------------


def to_examples(
    records: Iterable[FactCheckRecord],
    uid_split: dict[str, str] | None = None,
    text_cfg: TextConfig | None = None,
    require_image: bool = False,
    require_text: bool = True,
) -> list[Example]:
    """Convert records into examples, dropping what cannot be trained on.

    Drops, and why each one matters:
      * `unknown` label — not a negative, just unjudged
      * no text at all, when the text arm needs one
      * no usable image, when `require_image` (the fused arm needs both)
    """
    text_cfg = text_cfg or TextConfig()
    out: list[Example] = []
    dropped: Counter = Counter()

    for rec in records:
        if rec.label not in LABEL_TO_ID:
            dropped["unlabelled"] += 1
            continue

        image_path = pick_image(rec)
        if require_image and not image_path:
            dropped["no image"] += 1
            continue

        text = assemble_text(rec, text_cfg)
        if require_text and not text.strip():
            dropped["no text"] += 1
            continue

        out.append(Example(
            uid=rec.uid,
            text=text,
            label=LABEL_TO_ID[rec.label],
            image_path=image_path,
            source=rec.source,
            cluster_id=str(rec.raw.get("cluster_id") or rec.uid),
            split=(uid_split or {}).get(rec.uid, ""),
            image_only_claim=is_image_only_claim(rec),
            meta={"url": rec.url, "verdict": rec.verdict_raw,
                  "finance_score": rec.finance_score},
        ))

    if dropped:
        log.info("built %d examples, dropped %s", len(out), dict(dropped))
    return out


def load_examples(
    records_file: str | Path,
    manifest_file: str | Path | None = None,
    text_cfg: TextConfig | None = None,
    **kw,
) -> dict[str, list[Example]]:
    """Load curated records and return {split: [Example]}.

    Without a manifest every example lands in `train`, and that is loudly
    logged — training on an unsplit corpus is how a result gets reported that
    cannot be reproduced or trusted.
    """
    records = list(read_records(records_file))
    if not records:
        raise FileNotFoundError(
            f"no records in {records_file} — run scripts/04_curate.py first")

    uid_split: dict[str, str] = {}
    if manifest_file and Path(manifest_file).exists():
        from ..curation.splits import load_manifest

        uid_split, meta = load_manifest(manifest_file)
        if not meta.get("leakage_ok", False):
            raise ValueError(
                f"{manifest_file} records a FAILED leakage check. "
                "Every number produced from this split would be inflated. "
                "Re-run scripts/05_split.py before training.")
        log.info("split manifest %s (hash %s)", manifest_file,
                 meta.get("assignment_hash", "?"))
    else:
        log.warning("no split manifest — everything goes to 'train'. "
                    "Run scripts/05_split.py before reporting any number.")

    examples = to_examples(records, uid_split, text_cfg, **kw)
    by_split: dict[str, list[Example]] = {"train": [], "val": [], "test": []}
    for ex in examples:
        by_split.setdefault(ex.split or "train", []).append(ex)

    for name, items in by_split.items():
        labels = Counter(ID_TO_LABEL[e.label] for e in items)
        log.info("%-6s %5d examples  %s  (%d with image)",
                 name, len(items), dict(labels), sum(e.has_image for e in items))
    return by_split


def class_weights(examples: Sequence[Example]) -> dict[int, float]:
    """Inverse-frequency weights, normalised to mean 1.

    Fraud corpora are imbalanced and a plain cross-entropy loss will happily
    learn the majority class. Returns {} for a single-class set, where the
    concept does not apply and the real problem is Gate 2.
    """
    counts = Counter(e.label for e in examples)
    if len(counts) < 2:
        log.warning("only one class present (%s) — class weighting is meaningless "
                    "and no classifier can be trained; see CHECKLIST Gate 2",
                    dict(counts))
        return {}
    n, k = len(examples), len(counts)
    return {label: n / (k * c) for label, c in counts.items()}


def iter_batches(examples: Sequence[Example], size: int) -> Iterator[list[Example]]:
    """Plain batching, for evaluation paths that do not need a DataLoader."""
    if size < 1:
        raise ValueError("batch size must be >= 1")
    for i in range(0, len(examples), size):
        yield list(examples[i: i + size])
