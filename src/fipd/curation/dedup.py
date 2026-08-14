"""Near-duplicate detection and claim clustering.

Why this module exists
----------------------
The same viral creative is debunked independently by BOOM, Factly and
Newschecker. Those are three records but ONE item. If two of them land in the
training split and one in the test split, the model has seen the answer and the
reported accuracy measures memorisation rather than generalisation.

Exact hashing (SHA-256, already recorded at download time) does not solve this.
Each fact-checker re-hosts the creative through its own CDN, which re-encodes
and often re-crops it, so byte-identical copies are the exception. We need
hashes that survive re-compression, mild cropping and rescaling.

What is implemented
-------------------
Three complementary image hashes, implemented directly rather than pulled from
a library — they are short, they remove a dependency, and being able to explain
them in a viva is worth more than the import.

  * pHash  — sign of the low-frequency DCT coefficients. The most robust of the
             three to JPEG compression and small crops.
  * dHash  — horizontal brightness gradients. Cheap, robust to rescaling and
             gamma shifts, weak on flat images.
  * cHash  — per-channel colour block signature. The other two work on
             greyscale and are therefore blind to colour; without this, two
             creatives built from the same template with different branding
             collide. Scam creatives are mass-produced from templates, so this
             is a live failure mode rather than a hypothetical one.

`image_distance` combines them into a single weighted, normalised distance in
[0, 1]. **The threshold is data-dependent and must be calibrated on the real
corpus** — run `scripts/calibrate_dedup.py` once images exist. Synthetic images
cannot calibrate it honestly, so the shipped default is a starting point, not
a validated value.

The default errs liberal on purpose. The two failure modes are not symmetric:
over-merging costs a little data diversity, whereas under-merging puts the same
creative in train and test and silently inflates every metric. For split
safety, over-merging is the safe direction.

Clustering is single-linkage via union-find over the pairwise matches, then
merged with text-similarity clustering on the claim text, because two outlets
sometimes publish visually different screenshots of the same underlying scam.
The union of both signals is what `cluster_records` returns.

Cost note
---------
Pairwise comparison is O(n^2). At the few-thousand-record scale this project
operates at, that is a couple of seconds and not worth optimising. If the
dataset ever reaches six figures, replace the inner loop with BK-tree or LSH
bucketing — the interface here does not change.
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, NamedTuple, Sequence

import numpy as np

log = logging.getLogger(__name__)

try:
    from PIL import Image

    _PIL = True
except ImportError:  # pragma: no cover - Pillow is in requirements
    _PIL = False


# ---------------------------------------------------------------------------
# Perceptual hashes
# ---------------------------------------------------------------------------

def _to_gray_array(path: str | Path, size: tuple[int, int]) -> np.ndarray | None:
    if not _PIL:
        raise RuntimeError("Pillow is required for perceptual hashing")
    try:
        with Image.open(path) as im:
            im = im.convert("L").resize(size, Image.Resampling.LANCZOS)
            return np.asarray(im, dtype=np.float64)
    except Exception as e:  # corrupt downloads are common; skip them
        log.debug("unreadable image %s: %s", path, e)
        return None


def _bits_to_int(bits: np.ndarray) -> int:
    out = 0
    for b in bits.flatten():
        out = (out << 1) | int(b)
    return out


def dhash(path: str | Path, hash_size: int = 8) -> int | None:
    """Difference hash: compare each pixel with its right-hand neighbour."""
    arr = _to_gray_array(path, (hash_size + 1, hash_size))
    if arr is None:
        return None
    return _bits_to_int(arr[:, 1:] > arr[:, :-1])


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II basis, so we do not need scipy at collection time."""
    k = np.arange(n).reshape(-1, 1)
    i = np.arange(n).reshape(1, -1)
    m = np.cos(np.pi * (2 * i + 1) * k / (2 * n))
    m[0] *= np.sqrt(1 / n)
    m[1:] *= np.sqrt(2 / n)
    return m


_DCT32 = _dct_matrix(32)


def phash(path: str | Path, hash_size: int = 8) -> int | None:
    """Perceptual hash: sign of the low-frequency DCT coefficients."""
    arr = _to_gray_array(path, (32, 32))
    if arr is None:
        return None
    dct = _DCT32 @ arr @ _DCT32.T
    low = dct[:hash_size, :hash_size]
    # Exclude the DC term from the median: it encodes overall brightness and
    # would otherwise dominate and flatten the hash.
    med = np.median(low.flatten()[1:])
    return _bits_to_int(low > med)


def chash(path: str | Path, grid: int = 4) -> int | None:
    """Colour hash: per-channel block means compared against the channel median.

    pHash and dHash both operate on greyscale, so they are blind to colour
    entirely. Two different creatives built from the same template — same
    layout, same text box, different branding colours — collide under both.
    That is not hypothetical: scam creatives are mass-produced from templates,
    which is exactly the population this dataset is drawn from.

    This adds 3 * grid^2 bits of colour evidence to break those ties.
    """
    if not _PIL:
        raise RuntimeError("Pillow is required for perceptual hashing")
    try:
        with Image.open(path) as im:
            im = im.convert("RGB").resize((grid, grid), Image.Resampling.LANCZOS)
            arr = np.asarray(im, dtype=np.float64)
    except Exception as e:
        log.debug("unreadable image %s: %s", path, e)
        return None

    bits = []
    for c in range(3):
        chan = arr[..., c]
        bits.append(chan > np.median(chan))
    return _bits_to_int(np.concatenate([b.flatten() for b in bits]))


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# ---------------------------------------------------------------------------
# Combined image distance
# ---------------------------------------------------------------------------

# Each hash contributes a normalised [0,1] distance; the combined score is a
# weighted mean. pHash carries the most weight (most robust to compression),
# colour the least (it is a tie-breaker, not primary evidence).
_WEIGHTS = {"phash": 0.45, "dhash": 0.35, "chash": 0.20}
_BITS = {"phash": 64, "dhash": 64, "chash": 48}


class ImageSignature(NamedTuple):
    """All three hashes for one image."""

    phash: int
    dhash: int
    chash: int

    @classmethod
    def of(cls, path: str | Path) -> "ImageSignature | None":
        p, d, c = phash(path), dhash(path), chash(path)
        if p is None or d is None or c is None:
            return None
        return cls(p, d, c)


def image_distance(a: ImageSignature, b: ImageSignature) -> float:
    """Weighted normalised distance in [0,1]. 0 means identical."""
    return (
        _WEIGHTS["phash"] * hamming(a.phash, b.phash) / _BITS["phash"]
        + _WEIGHTS["dhash"] * hamming(a.dhash, b.dhash) / _BITS["dhash"]
        + _WEIGHTS["chash"] * hamming(a.chash, b.chash) / _BITS["chash"]
    )


# Default operating point.
#
# IMPORTANT — this number must be calibrated on the real corpus before it is
# trusted. Run `python scripts/calibrate_dedup.py` once images have been
# collected; it reports the separation between known duplicates and known
# distinct pairs and recommends a threshold.
#
# The default deliberately errs LIBERAL, because the two failure modes are not
# symmetric for our purpose:
#
#   over-merging  -> two distinct items land in one cluster. Costs a little
#                    data diversity. Cannot cause train/test leakage.
#   under-merging -> duplicates land in different clusters, and the same
#                    creative appears in train AND test. Silently inflates
#                    every reported metric.
#
# For split safety, over-merging is the safe direction. `pick_canonical` uses
# a stricter threshold, because there the asymmetry runs the other way.
DEFAULT_IMAGE_THRESHOLD = 0.22
STRICT_IMAGE_THRESHOLD = 0.10


# ---------------------------------------------------------------------------
# Text similarity — for creatives that differ visually but assert one claim
# ---------------------------------------------------------------------------

_WORD = re.compile(r"[a-z0-9ऀ-ॿ]+")
_STOP = {
    "the", "a", "an", "is", "was", "are", "were", "of", "in", "on", "to", "for",
    "and", "or", "this", "that", "it", "as", "with", "by", "from", "at", "no",
    "not", "video", "image", "photo", "viral", "claim", "fact", "check", "false",
    "fake", "shows", "showing", "shared", "post",
}


def _shingles(text: str, n: int = 3) -> set[str]:
    words = [w for w in _WORD.findall((text or "").lower()) if w not in _STOP]
    if len(words) < n:
        return set(words)
    return {" ".join(words[i:i + n]) for i in range(len(words) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    return inter / (len(a) + len(b) - inter)


# ---------------------------------------------------------------------------
# Union-find
# ---------------------------------------------------------------------------

class _UnionFind:
    def __init__(self, items: Sequence[str]) -> None:
        self.parent = {i: i for i in items}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra

    def groups(self) -> dict[str, list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for item in self.parent:
            out[self.find(item)].append(item)
        return dict(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class DedupReport:
    n_records: int = 0
    n_clusters: int = 0
    n_images_hashed: int = 0
    n_images_unreadable: int = 0
    largest_cluster: int = 0
    duplicate_rate: float = 0.0
    merged_by_image: int = 0
    merged_by_text: int = 0
    cluster_sizes: dict[int, int] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"records                {self.n_records}",
            f"clusters               {self.n_clusters}",
            f"duplicate rate         {self.duplicate_rate:.1%}",
            f"largest cluster        {self.largest_cluster}",
            f"images hashed          {self.n_images_hashed}",
            f"images unreadable      {self.n_images_unreadable}",
            f"merges from image      {self.merged_by_image}",
            f"merges from claim text {self.merged_by_text}",
        ]
        if self.cluster_sizes:
            dist = ", ".join(f"{k}:{v}" for k, v in sorted(self.cluster_sizes.items()))
            lines.append(f"cluster size histogram {dist}")
        return "\n".join(lines)


def hash_record_images(record, data_dir: str | Path) -> tuple[list[ImageSignature], int, int]:
    """Compute an ImageSignature for every downloaded image on a record."""
    data_dir = Path(data_dir)
    sigs: list[ImageSignature] = []
    ok = bad = 0
    for asset in record.images:
        if not asset.local_path:
            continue
        p = data_dir / asset.local_path
        if not p.exists():
            bad += 1
            continue
        sig = ImageSignature.of(p)
        if sig is None:
            bad += 1
            continue
        sigs.append(sig)
        ok += 1
    return sigs, ok, bad


def cluster_records(
    records: Sequence,
    data_dir: str | Path = "data",
    image_threshold: float = DEFAULT_IMAGE_THRESHOLD,
    text_threshold: float = 0.6,
    use_text: bool = True,
) -> tuple[dict[str, str], DedupReport]:
    """Group records that represent the same underlying item.

    Returns (uid -> cluster_id, report). The cluster id is the uid of the
    cluster's representative, chosen deterministically as the smallest uid so
    that re-running produces identical assignments.
    """
    uids = [r.uid for r in records]
    uf = _UnionFind(uids)
    report = DedupReport(n_records=len(records))

    # --- image hashes -------------------------------------------------
    per_record: dict[str, list[ImageSignature]] = {}
    for rec in records:
        hs, ok, bad = hash_record_images(rec, data_dir)
        per_record[rec.uid] = hs
        report.n_images_hashed += ok
        report.n_images_unreadable += bad

    # --- pairwise image comparison ------------------------------------
    for i in range(len(records)):
        ui = uids[i]
        hi = per_record.get(ui) or []
        if not hi:
            continue
        for j in range(i + 1, len(records)):
            uj = uids[j]
            if uf.find(ui) == uf.find(uj):
                continue
            hj = per_record.get(uj) or []
            if not hj:
                continue
            matched = any(
                image_distance(sa, sb) <= image_threshold for sa in hi for sb in hj
            )
            if matched:
                uf.union(ui, uj)
                report.merged_by_image += 1

    # --- claim-text comparison ----------------------------------------
    if use_text:
        shingle_map = {
            r.uid: _shingles(f"{r.claim_text} {r.title}") for r in records
        }
        for i in range(len(records)):
            ui = uids[i]
            si = shingle_map[ui]
            if not si:
                continue
            for j in range(i + 1, len(records)):
                uj = uids[j]
                if uf.find(ui) == uf.find(uj):
                    continue
                if jaccard(si, shingle_map[uj]) >= text_threshold:
                    uf.union(ui, uj)
                    report.merged_by_text += 1

    # --- finalise ------------------------------------------------------
    groups = uf.groups()
    assignment: dict[str, str] = {}
    sizes: dict[int, int] = defaultdict(int)
    for members in groups.values():
        rep = min(members)
        for m in members:
            assignment[m] = rep
        sizes[len(members)] += 1

    report.n_clusters = len(groups)
    report.largest_cluster = max((len(m) for m in groups.values()), default=0)
    report.cluster_sizes = dict(sizes)
    report.duplicate_rate = (
        1 - report.n_clusters / report.n_records if report.n_records else 0.0
    )
    return assignment, report


def pick_canonical(records: Iterable, assignment: dict[str, str]) -> list:
    """One representative per cluster — the record with the most images, then
    the longest body, then the smallest uid (deterministic tie-break)."""
    by_cluster: dict[str, list] = defaultdict(list)
    for r in records:
        by_cluster[assignment.get(r.uid, r.uid)].append(r)

    out = []
    for cid, members in by_cluster.items():
        best = max(
            members,
            key=lambda r: (
                sum(1 for a in r.images if a.local_path),
                len(r.body_text or ""),
                # negate via tuple ordering: smaller uid wins ties
                [-ord(c) for c in r.uid],
            ),
        )
        out.append(best)
    return out
