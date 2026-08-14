#!/usr/bin/env python3
"""Stage 07 — the ablation: text-only vs image-only vs fused.

This is the experiment Dr. Surati asked for, and the reason the model code
exists. It trains each arm at several seeds and reports mean ± std plus the
delta over the text-only baseline, because "fusion helps" is a claim about a
difference and three separate numbers do not state it.

    # everything from configs/model.yaml
    python scripts/07_ablation.py

    # one arm, one seed, while you are still debugging
    python scripts/07_ablation.py --arms fused --seeds 42 --epochs 1

    # the Gate 4 experiment: only posts whose caption is benign
    python scripts/07_ablation.py --image-only-claims

Outputs into data/processed/ablation/:
    <arm>_seed<seed>.json    per-run metrics, history, config, manifest hash
    ablation.json            everything, aggregated
    ablation.txt             the table for the report

Gate 4 is the one that matters: if `fused` does not beat `text_only` on posts
where the caption is benign and the claim is image-only, the central thesis is
not demonstrated regardless of overall F1.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.datasets.examples import class_weights, load_examples  # noqa: E402
from fipd.models.config import ARMS, load_model_config  # noqa: E402
from fipd.training.metrics import ablation_table  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--records", default=None,
                    help="default: <data-dir>/interim/finance.jsonl")
    ap.add_argument("--manifest", default=None,
                    help="default: <data-dir>/processed/split_manifest.json")
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--variant", default="student", choices=["student", "teacher"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--image-only-claims", action="store_true",
                    help="Gate 4: evaluate only where the caption is benign")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    log = setup(args.verbose)
    cfg = load_model_config(args.config)
    data_dir = Path(args.data_dir)

    records = Path(args.records) if args.records else data_dir / "interim" / "finance.jsonl"
    manifest = Path(args.manifest) if args.manifest else \
        data_dir / "processed" / "split_manifest.json"
    out_dir = Path(args.out_dir) if args.out_dir else data_dir / "processed" / "ablation"
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = args.arms or cfg.arms
    seeds = args.seeds or cfg.training.seeds
    epochs = args.epochs or cfg.training.epochs
    batch_size = args.batch_size or cfg.training.batch_size
    arch = getattr(cfg, args.variant)

    banner(f"Stage 07 — ablation ({args.variant}): {', '.join(arms)} x {len(seeds)} seeds")

    # ---- data (no torch needed yet, so config errors surface early) --------
    splits = load_examples(records, manifest, cfg.text)
    train_ex, val_ex, test_ex = splits["train"], splits["val"], splits["test"]

    if not train_ex or not val_ex:
        log.error("need a non-empty train and val split; got %d/%d. "
                  "Run scripts/04_curate.py and scripts/05_split.py first.",
                  len(train_ex), len(val_ex))
        return 1

    weights = class_weights(train_ex)
    if not weights:
        log.error("training data has ONE class only. No classifier can be trained "
                  "on this — see docs/CHECKLIST.md Gate 2 (negative-class strategy).")
        return 1

    if args.image_only_claims:
        before = len(test_ex)
        test_ex = [e for e in test_ex if e.image_only_claim]
        log.info("Gate 4 subset: %d of %d test examples have a benign caption",
                 len(test_ex), before)
        if len(test_ex) < 30:
            log.warning("only %d image-only-claim examples — too few to conclude "
                        "anything; report the count alongside any number", len(test_ex))

    # ---- torch from here --------------------------------------------------
    try:
        from fipd.datasets.torch_data import build_dataset, build_loader
        from fipd.models.arms import build_arm
        from fipd.training.loop import RunResult, evaluate_arm, train_arm
    except ImportError as e:
        log.error("%s", e)
        log.error('install the model extras: pip install -e ".[model]"')
        return 1

    manifest_hash = ""
    if manifest.exists():
        manifest_hash = json.loads(manifest.read_text(encoding="utf-8")).get(
            "assignment_hash", "")

    results: dict[str, list] = {arm: [] for arm in arms}

    for arm in arms:
        needs_image = arm in ("image_only", "fused")

        # The fused arm can only learn from items that have both modalities.
        # Restricting every arm to the same items keeps the comparison fair —
        # otherwise the arms are scored on different test sets.
        usable_train = [e for e in train_ex if e.has_image or not needs_image]
        usable_val = [e for e in val_ex if e.has_image or not needs_image]
        usable_test = [e for e in test_ex if e.has_image or not needs_image]
        log.info("[%s] %d train / %d val / %d test examples",
                 arm, len(usable_train), len(usable_val), len(usable_test))

        for seed in seeds:
            log.info("=" * 62)
            log.info("arm=%s seed=%d", arm, seed)
            log.info("=" * 62)

            ds_kw = dict(tokenizer_name=arch.text_encoder, image_root=data_dir,
                         image_size=cfg.measurement.image_size,
                         max_tokens=cfg.measurement.max_text_tokens,
                         needs_image=needs_image)
            train_loader = build_loader(
                build_dataset(usable_train, train=True, **ds_kw),
                batch_size=batch_size, shuffle=True,
                num_workers=cfg.training.num_workers, seed=seed)
            val_loader = build_loader(build_dataset(usable_val, **ds_kw),
                                      batch_size=batch_size)
            test_loader = build_loader(build_dataset(usable_test, **ds_kw),
                                       batch_size=batch_size)

            model = build_arm(arm, arch)
            run: RunResult = train_arm(
                model, train_loader, val_loader,
                arm=arm, seed=seed, epochs=epochs,
                lr=cfg.training.lr, head_lr=cfg.training.head_lr,
                weight_decay=cfg.training.weight_decay,
                warmup_ratio=cfg.training.warmup_ratio,
                max_grad_norm=cfg.training.max_grad_norm,
                patience=cfg.training.early_stopping_patience,
                class_weights=weights,
                checkpoint_dir=data_dir / "checkpoints",
                manifest_hash=manifest_hash,
            )

            if usable_test:
                run.test, _, _ = evaluate_arm(model, test_loader, arm,
                                              threshold=run.decision_threshold)
                log.info("[%s seed=%d] TEST %s", arm, seed, run.test.summary())

            run.save(out_dir / f"{arm}_seed{seed}.json")
            results[arm].append(run.test or run.val)

    # ---- report -----------------------------------------------------------
    metric = "macro_f1"
    table = ablation_table({k: [m for m in v if m] for k, v in results.items()},
                           metric=metric, reference="text_only")
    print("\n" + table + "\n")

    (out_dir / "ablation.txt").write_text(table, encoding="utf-8")
    (out_dir / "ablation.json").write_text(json.dumps(
        {arm: [m.to_dict() for m in runs if m] for arm, runs in results.items()},
        indent=2, default=str), encoding="utf-8")

    print(f"wrote {out_dir/'ablation.txt'} and {out_dir/'ablation.json'}")
    if args.image_only_claims:
        print("\nThis is the Gate 4 table. Quote the subset size next to it.\n")
    else:
        print("\nNow run with --image-only-claims. That subset is the thesis.\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
