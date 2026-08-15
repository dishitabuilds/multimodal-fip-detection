"""Deduplication, splitting and OCR-interface tests — fully offline.

Synthetic images are generated with PIL so the perceptual-hash tests exercise
real image transforms (JPEG re-compression, rescaling, cropping, brightness
shifts) rather than mocks. That matters: the whole point of pHash/dHash is
surviving exactly those transforms, and a mock would prove nothing.
"""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np  # noqa: E402
from PIL import Image, ImageDraw, ImageEnhance  # noqa: E402

from fipd.curation.dedup import (  # noqa: E402
    DEFAULT_IMAGE_THRESHOLD, ImageSignature, chash, cluster_records, dhash,
    hamming, image_distance, jaccard, phash, pick_canonical, _shingles,
)
from fipd.curation.splits import (  # noqa: E402
    load_manifest, save_manifest, split_by_cluster, verify_no_leakage,
)
from fipd.enrichment.ocr import BACKENDS, OCRCache, OCREngine, OCRResult, get_engine  # noqa: E402
from fipd.enrichment.translit import (  # noqa: E402
    build_text_input, is_code_mixed, normalise_mixed, normalise_unicode,
    romanised_to_devanagari,
)
from fipd.schema.records import FactCheckRecord, ImageAsset  # noqa: E402
from fipd.utils.logging_setup import use_utf8_stdout  # noqa: E402

use_utf8_stdout()  # these tests print Devanagari; a cp1252 console would crash

PASS = FAIL = 0


def check(cond, msg, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}   {detail}")


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

_PALETTE = [(200, 40, 40), (40, 120, 200), (30, 160, 90), (230, 170, 20),
            (140, 60, 190), (0, 150, 150), (220, 90, 40), (60, 60, 60)]


def make_creative(path: Path, seed: int, text: str = "GUARANTEED 40% RETURN") -> Path:
    """A deterministic pseudo-'scam creative'.

    Structured rather than random: a coloured background with a low-frequency
    gradient, a white claim box and a coloured badge. Different seeds produce
    genuinely different images in layout AND palette, which is what makes the
    'distinct images are not merged' assertion meaningful. Pure noise would be
    pathological for perceptual hashing and would not resemble real creatives.
    """
    col = _PALETTE[seed % len(_PALETTE)]
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    base[:, :] = col
    yy, xx = np.mgrid[0:256, 0:256]
    mode = seed % 3
    if mode == 0:
        base[..., 0] = np.clip(base[..., 0].astype(int) + yy // 2, 0, 255)
    elif mode == 1:
        base[..., 1] = np.clip(base[..., 1].astype(int) + xx // 2, 0, 255)
    else:
        base[..., 2] = np.clip(base[..., 2].astype(int) + (xx + yy) // 3, 0, 255)

    im = Image.fromarray(base)
    d = ImageDraw.Draw(im)
    x0, y0 = 20 + (seed * 17) % 70, 30 + (seed * 29) % 100
    d.rectangle([x0, y0, x0 + 150, y0 + 50], fill=(250, 250, 250))
    d.text((x0 + 8, y0 + 18), text, fill=(10, 10, 10))
    d.ellipse([30 + (seed * 23) % 110, 180, 110 + (seed * 23) % 110, 240],
              fill=_PALETTE[(seed + 3) % len(_PALETTE)])
    path.parent.mkdir(parents=True, exist_ok=True)
    im.save(path, quality=95)
    return path


def recompress(src: Path, dst: Path, quality: int = 35) -> Path:
    Image.open(src).save(dst, "JPEG", quality=quality)
    return dst


def rescale(src: Path, dst: Path, factor: float = 0.55) -> Path:
    im = Image.open(src)
    w, h = int(im.width * factor), int(im.height * factor)
    im.resize((w, h), Image.Resampling.LANCZOS).save(dst, quality=90)
    return dst


def crop_edges(src: Path, dst: Path, px: int = 10) -> Path:
    im = Image.open(src)
    im.crop((px, px, im.width - px, im.height - px)).save(dst, quality=90)
    return dst


def brighten(src: Path, dst: Path, factor: float = 1.25) -> Path:
    ImageEnhance.Brightness(Image.open(src)).enhance(factor).save(dst, quality=90)
    return dst


def rec(uid_seed: str, source: str, label: str, claim: str,
        images: list[str] | None = None, body: str = "") -> FactCheckRecord:
    r = FactCheckRecord(source=source, source_id=uid_seed, url=f"https://x/{uid_seed}",
                        title=claim, claim_text=claim, body_text=body or claim,
                        label=label)
    r.images = [ImageAsset(url=f"https://x/{p}", local_path=p, role="creative")
                for p in (images or [])]
    return r


# ---------------------------------------------------------------------------


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="fipd-curation-"))
    img_dir = tmp / "images"
    print(f"\nworkspace: {tmp}\n")

    # =================================================================
    print("[1] Perceptual hashing — invariance to real transforms")
    a = make_creative(img_dir / "a.jpg", seed=1)
    b = make_creative(img_dir / "b.jpg", seed=5, text="MUTUAL FUNDS SAHI HAI")

    pa, da, ca = phash(a), dhash(a), chash(a)
    check(all(isinstance(x, int) for x in (pa, da, ca)), "all three hashes are ints")
    check(0 <= pa < 2 ** 64 and 0 <= da < 2 ** 64, "grey hashes are 64-bit", (pa, da))
    check(0 <= ca < 2 ** 48, "colour hash is 48-bit", ca)
    check((phash(a), dhash(a), chash(a)) == (pa, da, ca), "hashing is deterministic")

    sig_a, sig_b = ImageSignature.of(a), ImageSignature.of(b)
    check(image_distance(sig_a, sig_a) == 0.0, "distance to self is zero")
    check(0.0 <= image_distance(sig_a, sig_b) <= 1.0, "distance is normalised to [0,1]",
          image_distance(sig_a, sig_b))

    variants = {
        "re-compressed q25": recompress(a, img_dir / "a_q25.jpg", quality=25),
        "rescaled 50%": rescale(a, img_dir / "a_small.jpg", factor=0.5),
        "cropped 12px": crop_edges(a, img_dir / "a_crop.jpg", px=12),
        "brightened 1.3x": brighten(a, img_dir / "a_bright.jpg", factor=1.3),
        "upscaled 2x": rescale(a, img_dir / "a_big.jpg", factor=2.0),
    }
    dup_dists = {}
    for name, v in variants.items():
        d = image_distance(sig_a, ImageSignature.of(v))
        dup_dists[name] = d
        check(d <= DEFAULT_IMAGE_THRESHOLD,
              f"near-duplicate survives {name}", f"distance={d:.3f}")

    # The invariant that genuinely matters and is threshold-independent:
    # every true duplicate must be closer than any distinct image.
    worst_dup = max(dup_dists.values())
    dist_pairs = [
        image_distance(ImageSignature.of(make_creative(img_dir / f"d{i}.jpg", seed=i)),
                       ImageSignature.of(make_creative(img_dir / f"d{j}.jpg", seed=j)))
        for i, j in [(1, 5), (2, 6), (3, 7), (0, 4)]
    ]
    check(worst_dup < min(dist_pairs),
          "every duplicate is closer than every distinct pair (separation holds)",
          f"worst dup={worst_dup:.3f}  closest distinct={min(dist_pairs):.3f}")
    check(min(dist_pairs) > DEFAULT_IMAGE_THRESHOLD,
          "distinct images sit above the default threshold",
          f"closest distinct={min(dist_pairs):.3f} vs t={DEFAULT_IMAGE_THRESHOLD}")

    check(phash(img_dir / "does_not_exist.jpg") is None, "missing file returns None")
    check(ImageSignature.of(img_dir / "nope.jpg") is None, "signature of missing file is None")

    bad = img_dir / "corrupt.jpg"
    bad.write_bytes(b"this is not an image")
    check(ImageSignature.of(bad) is None, "corrupt file returns None, does not raise")

    # =================================================================
    print("\n[2] Text shingling")
    s1 = _shingles("Viral post claims SEBI registered app gives guaranteed monthly returns")
    s2 = _shingles("Fake video claims SEBI registered app gives guaranteed monthly returns")
    s3 = _shingles("Old communal video from Bangladesh shared with false claim in India")
    check(jaccard(s1, s2) >= 0.6, "near-identical claims score high", round(jaccard(s1, s2), 3))
    check(jaccard(s1, s3) < 0.2, "unrelated claims score low", round(jaccard(s1, s3), 3))
    check(jaccard(set(), s1) == 0.0, "empty shingle set is safe")

    # =================================================================
    print("\n[3] Clustering across sources")
    records = [
        rec("boom1", "gfc:boomlive.in", "fake", "SEBI registered app promises guaranteed returns",
            ["images/a.jpg"]),
        rec("factly1", "gfc:factly.in", "fake", "SEBI registered app promises guaranteed returns",
            ["images/a_q25.jpg"]),                       # same creative, re-encoded
        rec("nc1", "gfc:newschecker.in", "fake", "SEBI registered app promises guaranteed returns",
            ["images/a_crop.jpg"]),                      # same creative, cropped
        rec("boom2", "gfc:boomlive.in", "real", "Mutual funds campaign is authentic",
            ["images/b.jpg"]),                           # different item
    ]
    assignment, dreport = cluster_records(records, data_dir=tmp,
                                          image_threshold=DEFAULT_IMAGE_THRESHOLD,
                                          text_threshold=0.6)
    check(len(set(assignment.values())) == 2, "3 duplicates + 1 unique -> 2 clusters",
          dreport.cluster_sizes)
    check(assignment[records[0].uid] == assignment[records[1].uid]
          == assignment[records[2].uid], "all three duplicates share a cluster")
    check(assignment[records[3].uid] != assignment[records[0].uid],
          "the distinct item is its own cluster")
    check(dreport.n_records == 4 and dreport.largest_cluster == 3, "report counts correct",
          dreport.summary())
    check(abs(dreport.duplicate_rate - 0.5) < 1e-9, "duplicate rate = 50%",
          dreport.duplicate_rate)
    check(dreport.n_images_hashed == 4, "all images hashed", dreport.n_images_hashed)

    a2, r2 = cluster_records(records, data_dir=tmp)
    check(a2 == assignment, "clustering is deterministic across runs")

    canon = pick_canonical(records, assignment)
    check(len(canon) == 2, "one canonical record per cluster", len(canon))

    no_text, _ = cluster_records(records, data_dir=tmp, use_text=False)
    check(len(set(no_text.values())) == 2, "image signal alone is sufficient here")

    # A threshold in the wrong units is the failure that actually happened:
    # 04_curate.py passed a raw Hamming distance (8) where a normalised one
    # is expected, and since image_distance() maxes out at 1.0, every pair
    # matched and the whole corpus became a single cluster.
    collapsed, _ = cluster_records(records, data_dir=tmp, image_threshold=8.0)
    check(len(set(collapsed.values())) == 1,
          "a threshold above 1.0 collapses everything — the bug this reproduces")
    check(DEFAULT_IMAGE_THRESHOLD < 1.0,
          "the real threshold is a normalised distance, not Hamming bits")

    # =================================================================
    print("\n[3b] Boilerplate images must not chain records together")
    # Site furniture — a masthead repeated on every article. Merging is
    # single-linkage, so one shared image chains two records, and chains are
    # transitive: without this filter a whole corpus of unrelated claims
    # collapses into one cluster. Observed for real on the first image run.
    shutil.copy(img_dir / "b.jpg", img_dir / "masthead.jpg")
    boiler = []
    for i in range(12):
        # Twelve unrelated claims, each carrying its own creative AND the
        # same masthead.
        src = "a.jpg" if i == 0 else "b.jpg"
        shutil.copy(img_dir / src, img_dir / f"distinct_{i}.jpg")
        boiler.append(rec(f"bp{i}", f"gfc:pub{i}.in", "fake",
                          f"Completely unrelated claim number {i} about topic {i}",
                          [f"images/distinct_{i}.jpg", "images/masthead.jpg"]))

    with_filter, rep_on = cluster_records(boiler, data_dir=tmp, use_text=False)
    without_filter, rep_off = cluster_records(boiler, data_dir=tmp, use_text=False,
                                              drop_boilerplate=False)
    check(len(set(without_filter.values())) == 1,
          "without the filter, the shared masthead collapses all 12 into one")
    check(len(set(with_filter.values())) > 1,
          "with the filter, they stay separate",
          f"got {len(set(with_filter.values()))} clusters")
    check(rep_on.n_boilerplate_signatures >= 1, "the boilerplate signature is reported")
    check(rep_off.n_boilerplate_signatures == 0, "and not counted when disabled")

    # It must stay inert on a small corpus, where a repeated image is far more
    # likely to be a genuine duplicate than a template.
    _, small = cluster_records(records, data_dir=tmp)
    check(small.n_boilerplate_signatures == 0,
          "no boilerplate is claimed on a 4-record corpus")

    # =================================================================
    print("\n[4] Group-aware splitting")
    many = []
    for i in range(60):
        cluster = f"c{i // 2}"          # 30 clusters of 2 records each
        for j in range(2):
            r = rec(f"r{i}_{j}", f"gfc:site{i % 3}.in",
                    "fake" if i % 4 else "real", f"claim number {i}")
            r.raw["cluster_id"] = cluster
            many.append(r)
    assign2 = {r.uid: r.raw["cluster_id"] for r in many}

    uid_split, sreport = split_by_cluster(many, assign2, ratios=(0.7, 0.15, 0.15), seed=42)
    ok, problems = verify_no_leakage(uid_split, assign2, many)
    check(ok, "no cluster spans two splits", problems[:3])
    check(sreport.leakage_ok, "report records the leakage check")
    check(set(uid_split.values()) == {"train", "val", "test"}, "all three splits populated",
          sreport.counts)
    check(sum(sreport.counts.values()) == len(many), "every record assigned", sreport.counts)
    check(sreport.n_clusters == 30, "cluster count correct", sreport.n_clusters)

    s2_split, _ = split_by_cluster(many, assign2, ratios=(0.7, 0.15, 0.15), seed=42)
    check(s2_split == uid_split, "same seed -> identical split")
    s3_split, _ = split_by_cluster(many, assign2, ratios=(0.7, 0.15, 0.15), seed=7)
    check(s3_split != uid_split, "different seed -> different split")

    for name in ("train", "val", "test"):
        labels = sreport.label_balance.get(name, {})
        check(len(labels) >= 1, f"'{name}' has labels recorded", labels)
    check(all(len(sreport.source_balance.get(n, {})) > 1 for n in ("train",)),
          "train draws from multiple sources", sreport.source_balance.get("train"))

    train_frac = sreport.counts["train"] / len(many)
    check(0.6 <= train_frac <= 0.8, "train share near requested 70%", round(train_frac, 3))

    # deliberate leakage must be caught
    bad = dict(uid_split)
    victim = many[0]
    sibling = next(r for r in many[1:] if assign2[r.uid] == assign2[victim.uid])
    bad[sibling.uid] = "test" if bad[victim.uid] != "test" else "train"
    ok_bad, probs = verify_no_leakage(bad, assign2, many)
    check(not ok_bad and probs, "injected leakage is detected", probs[:1])

    # bad ratios rejected
    for bad_ratio in [(0.5, 0.2, 0.2), (0.8, 0.3, -0.1)]:
        try:
            split_by_cluster(many, assign2, ratios=bad_ratio)
            check(False, f"ratios {bad_ratio} should raise")
        except ValueError:
            check(True, f"invalid ratios {bad_ratio} rejected")

    # tiny dataset still fills every split
    tiny = many[:6]
    tiny_assign = {r.uid: assign2[r.uid] for r in tiny}
    t_split, t_rep = split_by_cluster(tiny, tiny_assign, seed=1)
    check(verify_no_leakage(t_split, tiny_assign, tiny)[0], "tiny dataset has no leakage")
    check(sum(t_rep.counts.values()) == 6, "tiny dataset fully assigned", t_rep.counts)

    # =================================================================
    print("\n[5] Split manifest round-trip")
    mpath = save_manifest(uid_split, sreport, tmp / "processed", assignment=assign2)
    check(mpath.exists(), "manifest written")
    loaded, meta = load_manifest(mpath)
    check(loaded == uid_split, "manifest round-trips")
    check(meta["seed"] == 42 and meta["leakage_ok"] is True, "metadata preserved",
          {k: meta[k] for k in ("seed", "leakage_ok")})
    check(len(meta["assignment_hash"]) == 16, "assignment hash recorded",
          meta["assignment_hash"])

    # =================================================================
    print("\n[6] Transliteration and text assembly")
    check(romanised_to_devanagari("paisa double scheme") == "पैसा डबल स्कीम",
          "romanised finance terms mapped",
          romanised_to_devanagari("paisa double scheme"))
    check("निवेश" in romanised_to_devanagari("nivesh ghotaala"), "scam terms mapped")
    check(romanised_to_devanagari("quarterly earnings report")
          == "quarterly earnings report", "unknown English left untouched")
    check(is_code_mixed("SEBI registered निवेश स्कीम"), "code-mixing detected")
    check(not is_code_mixed("pure english text"), "pure Latin is not code-mixed")
    check(normalise_unicode("  a\n\n b  ") == "a b", "whitespace collapsed")

    mixed = normalise_mixed("paisa double guarantee", keep_original=True)
    check("paisa double" in mixed and "पैसा डबल" in mixed,
          "both surface forms kept", mixed)
    only = normalise_mixed("paisa double guarantee", keep_original=False)
    check("paisa" not in only and "पैसा" in only, "keep_original=False drops Latin", only)

    ti = build_text_input(caption="Check this out", ocr_text="GUARANTEED 40% RETURN",
                          title="Scam Alert")
    check(ti.index("GUARANTEED") < ti.index("Check this out"),
          "OCR text placed before caption (survives truncation)")
    check(build_text_input() == "", "empty inputs yield empty string")
    check(len(build_text_input(ocr_text="x" * 9000, max_chars=100)) <= 100,
          "max_chars enforced")

    # =================================================================
    print("\n[7] OCR interface")
    check(set(BACKENDS) == {"paddleocr", "easyocr", "tesseract"}, "three backends registered")
    try:
        get_engine("nope")
        check(False, "unknown backend should raise")
    except ValueError:
        check(True, "unknown backend rejected")

    class FakeEngine(OCREngine):
        name = "fake"

        def load(self):
            self.ready = True

        def _run(self, image_path):
            return (["GUARANTEED 40% MONTHLY", "सेबी रजिस्टर्ड"], [0.95, 0.81])

    eng = FakeEngine()
    res = eng.read(img_dir / "a.jpg")
    check(res.text.startswith("GUARANTEED"), "text joined from lines", res.text)
    check(abs(res.mean_confidence - 0.88) < 0.01, "mean confidence", res.mean_confidence)
    check(res.backend == "fake", "backend recorded")
    check(res.elapsed_ms >= 0, "timing recorded")
    sp = res.script_profile
    check(sp["devanagari"] > 0 and sp["latin"] > 0, "script profile detects code-mixing", sp)
    check(not res.looks_like_garbage(), "good text passes the quality gate")
    check(OCRResult(text="!!! ??").looks_like_garbage(), "punctuation soup flagged as garbage")
    check(OCRResult(text="ab").looks_like_garbage(), "too-short text flagged")

    class BrokenEngine(OCREngine):
        name = "broken"

        def load(self):
            raise RuntimeError("engine not installed")

        def _run(self, image_path):
            return [], []

    broken = BrokenEngine().read(img_dir / "a.jpg")
    check(broken.error and "not installed" in broken.error,
          "load failure returns an error result instead of raising", broken.error)

    class ExplodingEngine(OCREngine):
        name = "boom"

        def load(self):
            pass

        def _run(self, image_path):
            raise ValueError("corrupt image")

    exploded = ExplodingEngine().read(img_dir / "a.jpg")
    check(exploded.error == "corrupt image", "per-image failure is contained")

    # =================================================================
    print("\n[8] OCR cache and record integration")
    from fipd.enrichment.ocr import ocr_records

    cache_path = tmp / "ocr_cache.json"
    r_ocr = rec("ocr1", "gfc:boomlive.in", "fake", "claim", ["images/a.jpg", "images/b.jpg"])
    r_ocr.images[1].role = "annotated"

    stats = ocr_records([r_ocr], FakeEngine(), data_dir=tmp, cache_path=cache_path)
    check(stats["ocr_run"] == 1, "only the creative was OCR'd", stats)
    check(stats["skipped_annotated"] == 1, "verdict-stamp image skipped", stats)
    check(r_ocr.images[0].ocr_text.startswith("GUARANTEED"), "ocr_text written onto the asset")
    check(r_ocr.images[1].ocr_text is None, "annotated image left without ocr_text")
    check(cache_path.exists(), "cache flushed to disk")

    r2_ocr = rec("ocr1", "gfc:boomlive.in", "fake", "claim", ["images/a.jpg"])
    stats2 = ocr_records([r2_ocr], FakeEngine(), data_dir=tmp, cache_path=cache_path)
    check(stats2["cache_hits"] == 1 and stats2["ocr_run"] == 0, "second run hits the cache",
          stats2)

    cache = OCRCache(cache_path)
    check(cache.get("images/a.jpg") is not None, "cache reloads from disk")
    check(cache.get("nope") is None, "cache miss returns None")

    missing = rec("m1", "s", "fake", "c", ["images/gone.jpg"])
    st3 = ocr_records([missing], FakeEngine(), data_dir=tmp, cache_path=tmp / "c2.json")
    check(st3["missing"] == 1, "missing files counted, not crashed", st3)

    # =================================================================
    print("\n[9] Finance filter sees OCR text")
    from fipd.curation.finance_filter import score_record

    silent = rec("s1", "gfc:boomlive.in", "fake", "Check out this post")
    silent.images = [ImageAsset(url="u", local_path="p", role="creative")]
    before = score_record(silent).score
    silent.images[0].ocr_text = "SEBI REGISTERED guaranteed return paisa double"
    after = score_record(silent).score
    check(after > before, "OCR text raises the finance score", f"{before} -> {after}")
    check(score_record(silent).is_finance,
          "a benign caption with a scam image is now correctly flagged")

    shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 58)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 58 + "\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
