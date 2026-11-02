#!/usr/bin/env python3
"""Stage 07 — comparable, reproducible text/image/fused experiments.

    python scripts/07_ablation.py
    python scripts/07_ablation.py --arms fused --seeds 42 --epochs 1

Every invocation trains a fresh experiment unless --evaluate-only is supplied.
All arms use one verified multimodal cohort. Full-test predictions also score
an optional manually reviewed subset, without training a second model:

    python scripts/07_ablation.py --reviewed-subset data/reviewed_gate4.json
    python scripts/07_ablation.py --evaluate-only --experiment-id <printed-id> \
        --reviewed-subset data/reviewed_gate4.json

The subset JSON declares {"name": "gate4", "reviewed": true, "uids": [...]}.
It must contain only members of the held-out test cohort and both classes.
Automated image_only_claim flags are a review aid, not reviewed annotations.

Reports: data/processed/ablation/<variant>/<experiment-id>/
Checkpoints: data/checkpoints/<variant>/<experiment-id>/<arm>_seed<seed>.pt
The demo's legacy deployment checkpoint is an explicitly selected artifact;
this experiment runner never overwrites it.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _bootstrap  # noqa: F401,E402

from fipd.datasets.examples import (  # noqa: E402
    INPUT_PIPELINE_VERSION, class_weights,
)
from fipd.models.config import ARMS, load_model_config  # noqa: E402
from fipd.datasets.experiments import (  # noqa: E402
    common_cohort, file_hash, fingerprint, load_experiment_inputs,
)
from fipd.training.metrics import ablation_table, evaluate  # noqa: E402
from fipd.utils.logging_setup import banner, setup  # noqa: E402


def reviewed_subset(path: Path, test_examples):
    """Require an explicit human-review declaration and exact held-out IDs."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("reviewed") is not True:
        raise ValueError("subset manifest must declare reviewed: true after manual review")
    uids = data.get("uids")
    if not isinstance(uids, list) or not uids or any(not isinstance(uid, str) for uid in uids):
        raise ValueError("reviewed subset must contain a non-empty list of UID strings")
    uid_set = set(uids)
    if len(uid_set) != len(uids):
        raise ValueError("reviewed subset contains duplicate UIDs")
    test_ids = {ex.uid for ex in test_examples}
    unknown = uid_set - test_ids
    if unknown:
        raise ValueError(f"reviewed subset contains UIDs outside the common test cohort: "
                         f"{sorted(unknown)}")
    indices = [i for i, ex in enumerate(test_examples) if ex.uid in uid_set]
    if {test_examples[i].label for i in indices} != {0, 1}:
        raise ValueError("reviewed subset must include both classes for binary evaluation")
    name = re.sub(r"[^a-zA-Z0-9_-]+", "-", str(data.get("name") or "gate4")).strip("-")
    name = (name or "gate4")[:48]
    ids = [test_examples[i].uid for i in indices]
    key = f"reviewed-{name}-{fingerprint(sorted(ids))[:12]}"
    return key, indices, {"name": data.get("name") or "gate4", "reviewed": True,
                          "manifest_sha256": file_hash(path), "uids": ids}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--records", default=None)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--arms", nargs="+", default=None, choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=None)
    ap.add_argument("--variant", default="student", choices=["student", "teacher"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--image-only-claims", action="store_true",
                    help="Gate 4 evaluation; requires --reviewed-subset")
    ap.add_argument("--reviewed-subset", type=Path, default=None,
                    help="JSON declaring reviewed: true and held-out uids")
    ap.add_argument("--evaluate-only", action="store_true",
                    help="reuse checkpoints from --experiment-id; do not train")
    ap.add_argument("--experiment-id", default=None)
    ap.add_argument("--out-dir", default=None, help="report root; variant/ID are appended")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    log = setup(args.verbose)
    try:
        cfg = load_model_config(args.config)
        data_dir = Path(args.data_dir)
        records = Path(args.records) if args.records else data_dir / "interim" / "finance.jsonl"
        manifest = Path(args.manifest) if args.manifest else \
            data_dir / "processed" / "split_manifest.json"
        arms, seeds = args.arms or cfg.arms, args.seeds or cfg.training.seeds
        if len(set(arms)) != len(arms) or len(set(seeds)) != len(seeds):
            raise ValueError("arms and seeds must not contain duplicates")
        epochs = args.epochs if args.epochs is not None else cfg.training.epochs
        batch_size = args.batch_size if args.batch_size is not None else cfg.training.batch_size
        if epochs < 1 or batch_size < 1:
            raise ValueError("epochs and batch size must be >= 1")
        if args.image_only_claims and not args.reviewed_subset:
            raise ValueError("--image-only-claims requires --reviewed-subset; automated flags "
                             "are only a proxy and must be manually reviewed")
        if args.evaluate_only and not args.experiment_id:
            raise ValueError("--evaluate-only requires the existing --experiment-id")
        experiment_id = args.experiment_id or (
            datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8])
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", experiment_id):
            raise ValueError("experiment ID must be a simple directory name (up to 96 characters)")
        arch = getattr(cfg, args.variant)
        splits, manifest_data = load_experiment_inputs(
            records, manifest, cfg.text,
            allow_internal_smoke=cfg.raw.get("purpose") == "internal_smoke_only")
        cohort, excluded, cohort_hash = common_cohort(splits, data_dir)
        cohort_ids = {split: [ex.uid for ex in examples] for split, examples in cohort.items()}
        train_ex, val_ex, test_ex = (cohort[split] for split in ("train", "val", "test"))
        effective_config = {"input_pipeline_version": INPUT_PIPELINE_VERSION,
                            "architecture": asdict(arch), "text": asdict(cfg.text),
                            "training": {**asdict(cfg.training), "epochs": epochs,
                                         "batch_size": batch_size},
                            "measurement": asdict(cfg.measurement),
                            "evaluation": cfg.evaluation, "source_config": cfg.raw}
        identity = {"variant": args.variant, "config": effective_config,
                    "cohort_hash": cohort_hash, "cohort_ids": cohort_ids,
                    "records_sha256": file_hash(records),
                    "manifest_sha256": file_hash(manifest),
                    "manifest_hash": manifest_data.get("assignment_hash", "")}
        config_hash = fingerprint(effective_config)
        selections = {"full": (list(range(len(test_ex))), {"reviewed": False,
                                                         "uids": cohort_ids["test"]})}
        if args.reviewed_subset:
            key, indices, details = reviewed_subset(args.reviewed_subset, test_ex)
            selections[key] = (indices, details)
            if len(indices) < 30:
                log.warning("reviewed subset has only %d examples; report its size and "
                            "avoid strong conclusions", len(indices))
        out_root = Path(args.out_dir) if args.out_dir else data_dir / "processed" / "ablation"
        out_dir = out_root / args.variant / experiment_id
        checkpoint_dir = data_dir / "checkpoints" / args.variant / experiment_id
        experiment_file = out_dir / "experiment.json"
        if args.evaluate_only:
            saved = json.loads(experiment_file.read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError("data/config/cohort differ from this experiment; use its original "
                                 "inputs and effective settings for evaluation-only")
            for arm in arms:
                for seed in seeds:
                    for path in (checkpoint_dir / f"{arm}_seed{seed}.pt",
                                 out_dir / "runs" / f"{arm}_seed{seed}.json"):
                        if not path.is_file():
                            raise FileNotFoundError(f"missing experiment artifact: {path}")
        elif out_dir.exists() or checkpoint_dir.exists():
            raise FileExistsError(f"experiment {experiment_id} already exists; use "
                                  "--evaluate-only or choose a new experiment ID")
    except (OSError, ValueError, KeyError) as exc:
        log.error("%s", exc)
        return 1

    try:
        import torch

        from fipd.datasets.torch_data import build_dataset, build_loader
        from fipd.models.arms import build_arm
        from fipd.training.loop import RunResult, evaluate_arm, set_seed, train_arm
    except ImportError as exc:
        log.error('%s; install model extras with pip install -e ".[model]"', exc)
        return 1

    banner(f"Stage 07 — {args.variant}/{experiment_id}: {', '.join(arms)} x {len(seeds)} seeds")
    if not args.evaluate_only:
        out_dir.mkdir(parents=True)
        experiment_file.write_text(json.dumps(
            {"experiment_id": experiment_id, "identity": identity,
             "config_hash": config_hash, "excluded_ids": excluded}, indent=2), encoding="utf-8")
    weights = class_weights(train_ex)
    results = {name: {arm: [] for arm in arms} for name in selections}
    for arm in arms:
        for seed in seeds:
            # Seed construction as well as training: the head/fusion are random.
            set_seed(seed)
            ds_kw = dict(tokenizer_name=arch.text_encoder, image_root=data_dir,
                         image_size=cfg.measurement.image_size,
                         max_tokens=cfg.measurement.max_text_tokens,
                         needs_image=arm in ("image_only", "fused"))
            # Reloading a checkpoint needs architecture/config only; downloading
            # pretrained weight files here would immediately discard them.
            model = build_arm(arm, arch, from_scratch=args.evaluate_only)
            run_path = out_dir / "runs" / f"{arm}_seed{seed}.json"
            if args.evaluate_only:
                run = RunResult.load(run_path)
                ckpt = torch.load(checkpoint_dir / f"{arm}_seed{seed}.pt",
                                  map_location="cpu", weights_only=True)
                for key, value in (("experiment_id", experiment_id), ("variant", args.variant),
                                   ("arm", arm), ("seed", seed), ("cohort_hash", cohort_hash)):
                    if ckpt.get(key) != value:
                        raise ValueError(f"checkpoint {key} does not match the experiment")
                model.load_state_dict(ckpt["state_dict"])
                run.decision_threshold = float(ckpt["threshold"])
            else:
                train_loader = build_loader(build_dataset(train_ex, train=True, **ds_kw),
                                            batch_size=batch_size, shuffle=True,
                                            num_workers=cfg.training.num_workers, seed=seed)
                val_loader = build_loader(build_dataset(val_ex, **ds_kw), batch_size=batch_size)
                run = train_arm(
                    model, train_loader, val_loader, arm=arm, seed=seed, epochs=epochs,
                    lr=cfg.training.lr, head_lr=cfg.training.head_lr,
                    weight_decay=cfg.training.weight_decay, warmup_ratio=cfg.training.warmup_ratio,
                    max_grad_norm=cfg.training.max_grad_norm,
                    patience=cfg.training.early_stopping_patience, class_weights=weights,
                    checkpoint_dir=checkpoint_dir, manifest_hash=identity["manifest_hash"],
                    variant=args.variant, experiment_id=experiment_id, cohort_hash=cohort_hash,
                    cohort_ids=cohort_ids, run_config=effective_config)

            test_loader = build_loader(build_dataset(test_ex, **ds_kw), batch_size=batch_size)
            run.test, labels, scores = evaluate_arm(model, test_loader, arm,
                                                   threshold=run.decision_threshold)
            if len(labels) != len(test_ex):
                raise ValueError("test prediction count differs from the recorded cohort")
            for name, (indices, details) in selections.items():
                metrics = evaluate(labels[indices], scores[indices],
                                   threshold=run.decision_threshold)
                ids = [test_ex[i].uid for i in indices]
                run.evaluation_ids[name] = ids
                run.evaluations[name] = metrics
                results[name][arm].append(metrics)
                eval_dir = out_dir / "evaluations" / name
                eval_dir.mkdir(parents=True, exist_ok=True)
                payload = {"experiment_id": experiment_id, "variant": args.variant,
                           "arm": arm, "seed": seed, "config_hash": config_hash,
                           "cohort_hash": cohort_hash, "evaluation": details,
                           "metrics": metrics.to_dict(), "checkpoint": run.checkpoint,
                           "predictions": [{"uid": test_ex[i].uid, "label": int(labels[i]),
                                            "score": float(scores[i])} for i in indices]}
                (eval_dir / f"{arm}_seed{seed}.json").write_text(
                    json.dumps(payload, indent=2), encoding="utf-8")
            run.save(run_path)

    # Selection hashes preserve reports from smaller arm/seed invocations too.
    selection_id = fingerprint({"arms": sorted(arms), "seeds": sorted(seeds)})[:12]
    for name, arm_results in results.items():
        table = ablation_table(arm_results, metric="macro_f1", reference="text_only")
        eval_dir = out_dir / "evaluations" / name
        base = eval_dir / f"ablation_{selection_id}"
        base.with_suffix(".txt").write_text(table, encoding="utf-8")
        base.with_suffix(".json").write_text(json.dumps(
            {"experiment_id": experiment_id, "variant": args.variant, "evaluation": name,
             "config_hash": config_hash, "cohort_hash": cohort_hash,
             "uids": selections[name][1]["uids"],
             "arms": {arm: [metric.to_dict() for metric in runs]
                      for arm, runs in arm_results.items()}, "seeds": seeds}, indent=2),
            encoding="utf-8")
        print(f"\n{name}: {len(selections[name][0])} held-out examples\n{table}\n")
    print(f"Experiment ID: {experiment_id}\nReports: {out_dir}\nCheckpoints: {checkpoint_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
