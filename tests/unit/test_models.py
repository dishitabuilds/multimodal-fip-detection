"""Budget, metrics and example-building — all offline, no torch required.

These are the parts of the model stack whose mistakes are silent. A wrong
PR-AUC, an example set that drops the minority class, or a budget check that
passes a model it should fail all produce plausible-looking numbers, so they are
tested against hand-computed values rather than against themselves.

Where sklearn happens to be installed, the metric implementations are also
cross-checked against it. That is the real test: two independent implementations
agreeing is evidence, one implementation agreeing with itself is not.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT_DIR / "src"))

import numpy as np  # noqa: E402

from fipd.datasets.examples import (  # noqa: E402
    Example, assemble_text, class_weights, is_image_only_claim, pick_image, to_examples,
)
from fipd.models.budget import (  # noqa: E402
    Budget, Measurement, check_budget, format_report, measure_latency, percentile,
    save_report,
)
from fipd.models.config import ARMS, TextConfig, load_model_config  # noqa: E402
from fipd.schema.records import FactCheckRecord, ImageAsset  # noqa: E402
from fipd.training.metrics import (  # noqa: E402
    ablation_table, aggregate, best_threshold, confusion, evaluate, macro_f1,
    pr_auc, pr_curve, precision_recall_f1, roc_auc,
)
from fipd.utils.logging_setup import use_utf8_stdout  # noqa: E402

use_utf8_stdout()

PASS = FAIL = 0


def check(cond, msg, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  PASS  {msg}")
    else:
        FAIL += 1
        print(f"  FAIL  {msg}   {detail}")


def close(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


def main() -> int:
    tmp = Path(tempfile.mkdtemp(prefix="mp-model-test-"))

    # ------------------------------------------------------------------
    print("\n[1] Confusion matrix and P/R/F1")
    # 10 items: 4 true positives, 1 false positive, 2 false negatives, 3 true negatives
    y_true = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    y_pred = [1, 1, 1, 1, 0, 0, 1, 0, 0, 0]
    cm = confusion(y_true, y_pred)
    check((cm.tp, cm.fp, cm.fn, cm.tn) == (4, 1, 2, 3), "counts correct",
          f"got tp={cm.tp} fp={cm.fp} fn={cm.fn} tn={cm.tn}")
    check(cm.total == 10, "total is the sample count")

    p, r, f1 = precision_recall_f1(cm)
    check(close(p, 4 / 5), f"precision = 4/5 (got {p:.4f})")
    check(close(r, 4 / 6), f"recall = 4/6 (got {r:.4f})")
    check(close(f1, 2 * (4 / 5) * (4 / 6) / ((4 / 5) + (4 / 6))), f"F1 (got {f1:.4f})")

    # The degenerate cases are the ones that matter: a model that never fires
    # must score 0, not raise and not return NaN.
    never = confusion([1, 1, 0, 0], [0, 0, 0, 0])
    p0, r0, f0 = precision_recall_f1(never)
    check((p0, r0, f0) == (0.0, 0.0, 0.0), "model that never predicts fake scores 0")
    check(not any(np.isnan([p0, r0, f0])), "no NaN leaks into a mean over seeds")

    # ------------------------------------------------------------------
    print("\n[2] macro-F1 weights the rare class equally")
    # 90 negatives, 10 positives; predict everything negative.
    yt = [1] * 10 + [0] * 90
    yp = [0] * 100
    m_cm = confusion(yt, yp)
    check(close(macro_f1(m_cm), (0.0 + 2 * (90 / 90) * (90 / 100) /
                                 ((90 / 90) + (90 / 100))) / 2),
          f"macro-F1 of an all-negative model on 10:90 (got {macro_f1(m_cm):.4f})")
    check(macro_f1(m_cm) < 0.5, "macro-F1 punishes it, unlike accuracy (0.90)")

    # ------------------------------------------------------------------
    print("\n[3] Ranking metrics")
    yt = [0, 0, 1, 1]
    ys = [0.1, 0.4, 0.35, 0.8]
    check(close(roc_auc(yt, ys), 0.75), f"ROC-AUC = 0.75 (got {roc_auc(yt, ys):.4f})")

    perfect = roc_auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    check(close(perfect, 1.0), f"perfect ranking -> 1.0 (got {perfect:.4f})")
    inverted = roc_auc([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1])
    check(close(inverted, 0.0), f"inverted ranking -> 0.0 (got {inverted:.4f})")
    tied = roc_auc([0, 1], [0.5, 0.5])
    check(close(tied, 0.5), f"all-tied scores -> 0.5 (got {tied:.4f})")

    ap = pr_auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    check(close(ap, 0.8333333333333333, tol=1e-6), f"average precision (got {ap:.6f})")
    check(pr_auc([1, 1, 1], [0.2, 0.5, 0.9]) == 0.0,
          "single-class input returns 0.0 rather than a fabricated number")

    prec, rec, thr = pr_curve([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8])
    check(len(prec) == len(rec) == len(thr), "PR curve arrays are the same length")
    check(bool(np.all(np.diff(rec) >= 0)), "recall is non-decreasing along the curve")
    check(close(rec[-1], 1.0), "the loosest threshold reaches recall 1.0")

    # ------------------------------------------------------------------
    print("\n[4] evaluate() and the baseline it must beat")
    yt = [1] * 80 + [0] * 20
    ys = [0.9] * 80 + [0.9] * 20      # flags everything
    m = evaluate(yt, ys, threshold=0.5)
    check(close(m.recall, 1.0), "flag-everything has perfect recall")
    check(close(m.precision, 0.8), "...and precision equal to the base rate")
    check(close(m.majority_baseline, 0.8), "majority baseline reported (0.8)")
    check(close(m.accuracy, 0.8), "accuracy equals the baseline — it says nothing")
    check(m.n == 100 and m.n_positive == 80, "sample counts recorded")

    t, v = best_threshold([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9], metric="f1")
    check(close(v, 1.0), f"a separable set reaches F1 1.0 at t={t:.2f}")

    try:
        evaluate([1, 0], [0.5])
        check(False, "length mismatch rejected")
    except ValueError:
        check(True, "length mismatch rejected")
    try:
        evaluate([0, 2], [0.1, 0.9])
        check(False, "non-binary labels rejected")
    except ValueError:
        check(True, "non-binary labels rejected")

    # ------------------------------------------------------------------
    print("\n[5] Cross-check against sklearn")
    try:
        from sklearn.metrics import (average_precision_score, f1_score,
                                     precision_score, recall_score, roc_auc_score)

        rng = np.random.default_rng(0)
        yt = rng.integers(0, 2, 200)
        ys = rng.random(200)
        # Make the scores carry real signal, not just noise.
        ys = np.clip(ys * 0.5 + yt * 0.4, 0, 1)
        mine = evaluate(yt, ys, threshold=0.5)
        yp = (ys >= 0.5).astype(int)

        check(close(mine.roc_auc, roc_auc_score(yt, ys), 1e-9),
              f"ROC-AUC matches sklearn ({mine.roc_auc:.6f})")
        check(close(mine.pr_auc, average_precision_score(yt, ys), 1e-9),
              f"PR-AUC matches sklearn ({mine.pr_auc:.6f})")
        check(close(mine.precision, precision_score(yt, yp, zero_division=0), 1e-9),
              "precision matches sklearn")
        check(close(mine.recall, recall_score(yt, yp, zero_division=0), 1e-9),
              "recall matches sklearn")
        check(close(mine.f1, f1_score(yt, yp, zero_division=0), 1e-9),
              "F1 matches sklearn")
        check(close(mine.macro_f1, f1_score(yt, yp, average="macro", zero_division=0), 1e-9),
              "macro-F1 matches sklearn")
    except ImportError:
        print("  SKIP  sklearn not installed — cross-check not run")

    # ------------------------------------------------------------------
    print("\n[6] Seed aggregation and the ablation table")
    runs = [evaluate([0, 0, 1, 1], [0.1, 0.2, 0.7, 0.9]),
            evaluate([0, 0, 1, 1], [0.1, 0.6, 0.4, 0.9])]
    agg = aggregate(runs, "macro_f1")
    check(agg.n_seeds == 2, "seed count recorded")
    check(close(agg.mean, float(np.mean([r.macro_f1 for r in runs]))), "mean correct")
    check(agg.std > 0, "std is non-zero when the seeds disagree")
    check(aggregate([runs[0]], "macro_f1").std == 0.0, "single seed has std 0, not NaN")

    table = ablation_table({"text_only": [runs[1]], "fused": [runs[0]]},
                           metric="macro_f1", reference="text_only")
    check("fused" in table and "text_only" in table, "both arms appear in the table")
    check("+" in table or "-" in table, "the delta column carries a sign")

    # ------------------------------------------------------------------
    print("\n[7] The lightweight budget")
    budget = Budget(max_params_m=50, max_disk_mb=100, max_latency_p50_ms=100,
                    max_latency_p95_ms=250, max_peak_ram_mb=1024,
                    min_f1_retention=0.95)

    good = Measurement(name="student", params_m=42.0, disk_mb=88.0,
                       latency_p50_ms=61.0, latency_p95_ms=140.0,
                       peak_ram_mb=700.0, f1=0.82, teacher_f1=0.84)
    checks = {c.name: c for c in check_budget(good, budget)}
    check(all(c.ok for c in checks.values()), "a model inside every target passes")
    check(close(checks["parameters"].headroom, 8.0), "headroom is limit minus measured")

    heavy = Measurement(name="teacher", params_m=330.0, disk_mb=1300.0,
                        latency_p50_ms=900.0, latency_p95_ms=1200.0,
                        peak_ram_mb=2600.0, f1=0.84, teacher_f1=0.84)
    hc = {c.name: c for c in check_budget(heavy, budget)}
    check(hc["parameters"].status == "OVER", "an over-budget model is marked OVER")
    check(hc["F1 retention"].ok, "the teacher trivially retains its own F1")

    # The row that stops the budget being gamed.
    fast_and_useless = Measurement(name="degenerate", params_m=1.0, disk_mb=4.0,
                                   latency_p50_ms=3.0, latency_p95_ms=5.0,
                                   peak_ram_mb=100.0, f1=0.40, teacher_f1=0.84)
    fc = {c.name: c for c in check_budget(fast_and_useless, budget)}
    check(all(fc[k].ok for k in ("parameters", "disk size", "latency p50")),
          "a tiny useless model passes the size and speed rows")
    check(fc["F1 retention"].status == "UNDER",
          "...and is caught by the F1-retention row, which is the point")

    unmeasured = Measurement(name="nothing")
    uc = check_budget(unmeasured, budget)
    check(all(c.status == "not measured" for c in uc),
          "an unmeasured model reports 'not measured', never a default pass")
    check(not any(c.ok for c in uc), "and no unmeasured row counts as a pass")

    report = format_report(good, budget)
    check("PASS" in report and "student" in report, "report renders")
    check("not measured" in format_report(unmeasured, budget),
          "unmeasured rows are visible in the report")
    p = save_report(good, budget, tmp / "budget.json")
    check(p.exists() and "checks" in p.read_text(encoding="utf-8"), "budget JSON written")

    # ------------------------------------------------------------------
    print("\n[8] Latency measurement")
    calls = {"n": 0}

    def counted():
        calls["n"] += 1

    timing = measure_latency(counted, warmup_runs=5, timed_runs=20)
    check(calls["n"] == 25, f"warm-up runs happen but are not timed (got {calls['n']})")
    check(timing["runs"] == 20, "timed run count recorded")
    check(timing["p50_ms"] <= timing["p95_ms"] <= timing["max_ms"],
          "p50 <= p95 <= max")
    check(timing["min_ms"] <= timing["p50_ms"], "min <= p50")

    check(close(percentile([1, 2, 3, 4], 50), 2.5), "percentile interpolates")
    check(close(percentile([5], 95), 5.0), "single sample percentile is that sample")
    try:
        measure_latency(counted, timed_runs=0)
        check(False, "timed_runs=0 rejected")
    except ValueError:
        check(True, "timed_runs=0 rejected")

    # ------------------------------------------------------------------
    print("\n[9] Model config")
    cfg = load_model_config(ROOT_DIR / "configs" / "model.yaml")
    check(cfg.teacher.text_encoder.startswith("google/muril"), "teacher text encoder read")
    check(all(a in ARMS for a in cfg.arms), "declared arms are valid")
    check(len(cfg.training.seeds) >= 3, "at least 3 seeds configured, so std means something")
    check(cfg.measurement.threads > 0 and cfg.measurement.timed_runs > 0,
          "measurement conditions present")
    check(cfg.text.fields[0] == "ocr_text",
          "OCR text is first, so truncation eats the caption not the claim")

    # Both variants must be buildable as configured. This is checkable without
    # torch and would otherwise surface as an opaque error deep inside
    # MultiheadAttention, minutes into a run.
    for variant in ("teacher", "student"):
        arch = getattr(cfg, variant)
        check(arch.hidden_dim % arch.fusion_heads == 0,
              f"{variant}: hidden_dim {arch.hidden_dim} divides by "
              f"fusion_heads {arch.fusion_heads}")

    try:
        from fipd.models.arms import build_arm

        build_arm("no_such_arm", cfg.student)
        check(False, "an unknown arm name is rejected")
    except ValueError:
        check(True, "an unknown arm name is rejected")
    except ImportError:
        print("  SKIP  torch not installed — arm construction not exercised")
    check("reference_machine" in str(cfg.measurement.reference_machine) or
          bool(cfg.measurement.reference_machine),
          "a reference machine is named — latency without one is not comparable")

    # ------------------------------------------------------------------
    print("\n[10] Records -> examples")
    rec_fake = FactCheckRecord(
        source="boomlive", source_id="1", url="https://x/1",
        title="Viral post claims SEBI approved this trading app",
        claim_text="Double your money in 30 days", verdict_raw="False", label="fake",
        images=[
            ImageAsset(url="https://x/creative.jpg", local_path="images/b/creative.jpg",
                       role="creative", width=1200, height=800,
                       ocr_text="GUARANTEED 300% RETURNS - SEBI REGISTERED"),
            ImageAsset(url="https://x/stamp.jpg", local_path="images/b/stamp.jpg",
                       role="annotated", width=2000, height=2000,
                       ocr_text="FALSE"),
        ])
    rec_real = FactCheckRecord(
        source="factly", source_id="2", url="https://x/2",
        title="AMFI mutual fund awareness campaign is genuine",
        claim_text="This AMFI advert is authentic", verdict_raw="True", label="real",
        images=[ImageAsset(url="https://x/amfi.jpg", local_path="images/f/amfi.jpg",
                           role="creative", width=800, height=600,
                           ocr_text="Mutual funds are subject to market risks")])
    rec_unknown = FactCheckRecord(source="altnews", source_id="3", url="https://x/3",
                                  title="Unverified", label="unknown")

    chosen = pick_image(rec_fake)
    check(chosen == "images/b/creative.jpg",
          "the creative is chosen over the fact-checker's verdict stamp",
          f"got {chosen}")
    check(pick_image(rec_unknown) is None, "a record with no images yields None")

    text = assemble_text(rec_fake, TextConfig())
    check("300%" in text or "GUARANTEED" in text, "OCR text reaches the text input")
    check(text.index("GUARANTEED") < (text.index("Double") if "Double" in text else 10**9),
          "OCR text comes before the caption")

    examples = to_examples([rec_fake, rec_real, rec_unknown],
                           {rec_fake.uid: "train", rec_real.uid: "test"})
    check(len(examples) == 2, "the unlabelled record is dropped, not treated as real",
          f"got {len(examples)}")
    by_uid = {e.uid: e for e in examples}
    check(by_uid[rec_fake.uid].label == 1, "fake maps to 1")
    check(by_uid[rec_real.uid].label == 0, "real maps to 0")
    check(by_uid[rec_fake.uid].split == "train", "split carried through from the manifest")
    check(by_uid[rec_real.uid].split == "test", "test split carried through")
    check(by_uid[rec_fake.uid].has_image, "image path attached")

    # Gate 4 subset. The caption side is scored on claim_text alone; the
    # fact-checker's own headline names the scam in nearly every record, so
    # including it would leave this subset permanently empty.
    benign_caption = FactCheckRecord(
        source="boomlive", source_id="5", url="https://x/5", label="fake",
        title="Viral post falsely claims SEBI approved this trading app",
        claim_text="Look at this screenshot my friend sent",
        images=[ImageAsset(url="u", local_path="p", role="creative",
                           ocr_text="GUARANTEED 300% RETURNS SEBI REGISTERED "
                                    "INVESTMENT TRADING PROFIT DEMAT")])
    check(is_image_only_claim(benign_caption),
          "a benign caption with a loud scam image is an image-only claim")
    check(not is_image_only_claim(rec_fake),
          "a caption that already names the scam is NOT an image-only claim")

    no_ocr = FactCheckRecord(source="s", source_id="4", url="u", label="fake",
                             title="SEBI investment scam", claim_text="invest now",
                             images=[ImageAsset(url="u", local_path="p", role="creative")])
    check(not is_image_only_claim(no_ocr),
          "no OCR text means it cannot be an image-only claim")

    # ------------------------------------------------------------------
    print("\n[11] Class weights")
    imbalanced = ([Example(uid=f"f{i}", text="t", label=1) for i in range(90)]
                  + [Example(uid=f"r{i}", text="t", label=0) for i in range(10)])
    w = class_weights(imbalanced)
    check(w[0] > w[1], "the rare class gets the larger weight",
          f"got {w}")
    check(close(w[1], 100 / (2 * 90)) and close(w[0], 100 / (2 * 10)),
          "inverse-frequency weights are correct")
    check(class_weights([Example(uid="a", text="t", label=1)]) == {},
          "a single-class set returns no weights instead of a meaningless one")

    # ------------------------------------------------------------------
    print("\n" + "=" * 58)
    print(f"  {PASS} passed, {FAIL} failed")
    print("=" * 58 + "\n")
    return 1 if FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
