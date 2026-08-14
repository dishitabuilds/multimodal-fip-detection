"""Training and evaluation loop for one ablation arm at one seed.

Deliberately plain: no Trainer subclass, no callback framework. The things that
actually decide whether a result is believable — the seed, the class weights,
the early-stopping criterion, which split the threshold was picked on — are all
visible in one file rather than distributed across a framework's hooks.

Every run records its seed, its config and the split-manifest hash it trained
under, because the reproducibility appendix (docs/CHECKLIST.md Phase 5) needs
them and reconstructing them afterwards is impossible.
"""

from __future__ import annotations

import json
import logging
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..models.encoders import require_torch
from .metrics import Metrics, best_threshold, evaluate

log = logging.getLogger(__name__)


@dataclass
class RunResult:
    """Everything needed to report — and to reproduce — one training run."""

    arm: str
    seed: int
    epochs_run: int = 0
    best_epoch: int = 0
    val: Metrics | None = None
    test: Metrics | None = None
    decision_threshold: float = 0.5
    train_seconds: float = 0.0
    manifest_hash: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    history: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["val"] = self.val.to_dict() if self.val else None
        d["test"] = self.test.to_dict() if self.test else None
        return d

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2, default=str),
                        encoding="utf-8")
        return path


def set_seed(seed: int) -> None:
    """Seed every RNG that touches a result.

    cudnn determinism is left ON at the cost of some speed. A run that cannot be
    reproduced cannot be defended in a viva.
    """
    torch = require_torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _forward(model, batch, device, arm: str):
    """Feed a batch to an arm, passing only the modalities it accepts."""
    kw: dict[str, Any] = {}
    if arm in ("text_only", "fused"):
        kw["input_ids"] = batch["input_ids"].to(device)
        kw["attention_mask"] = batch["attention_mask"].to(device)
    if arm in ("image_only", "fused"):
        kw["pixel_values"] = batch["pixel_values"].to(device)
    return model(**kw)


def predict(model, loader, device, arm: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (labels, positive-class probabilities) for a whole loader."""
    torch = require_torch()
    model.eval()
    labels: list[int] = []
    scores: list[float] = []

    with torch.no_grad():
        for batch in loader:
            logits = _forward(model, batch, device, arm)
            probs = torch.softmax(logits, dim=-1)[:, 1]
            scores.extend(probs.detach().cpu().tolist())
            labels.extend(batch["label"].tolist())

    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def train_arm(
    model,
    train_loader,
    val_loader,
    *,
    arm: str,
    seed: int,
    epochs: int = 8,
    lr: float = 2e-5,
    head_lr: float = 1e-4,
    weight_decay: float = 0.01,
    warmup_ratio: float = 0.1,
    max_grad_norm: float = 1.0,
    patience: int = 3,
    class_weights: dict[int, float] | None = None,
    device: str | None = None,
    checkpoint_dir: str | Path | None = None,
    manifest_hash: str = "",
) -> RunResult:
    """Fine-tune one arm, early-stopping on validation macro-F1.

    macro-F1 rather than loss or accuracy: on imbalanced data, validation loss
    can improve while the minority class gets worse, and that is the class the
    whole project is about.
    """
    torch = require_torch()
    import torch.nn as nn

    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # The randomly-initialised head can take a larger step than a pretrained
    # backbone, which only needs nudging. One LR for both either cooks the
    # encoder or leaves the head undertrained.
    head_params, backbone_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head_params if name.startswith(("classifier", "fusion", "text_proj",
                                         "image_proj")) else backbone_params).append(p)

    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": lr},
         {"params": head_params, "lr": head_lr}],
        weight_decay=weight_decay,
    )

    total_steps = max(1, len(train_loader) * epochs)
    warmup_steps = int(total_steps * warmup_ratio)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    weight_tensor = None
    if class_weights:
        weight_tensor = torch.tensor(
            [class_weights.get(0, 1.0), class_weights.get(1, 1.0)],
            dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight_tensor)

    result = RunResult(arm=arm, seed=seed, manifest_hash=manifest_hash,
                       config={"lr": lr, "head_lr": head_lr, "epochs": epochs,
                               "weight_decay": weight_decay,
                               "class_weights": class_weights or {}})
    best_score, best_state, epochs_without_gain = -1.0, None, 0
    started = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = _forward(model, batch, device, arm)
            loss = criterion(logits, batch["label"].to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            running += float(loss.item())

        y_true, y_score = predict(model, val_loader, device, arm)
        val_metrics = evaluate(y_true, y_score)
        train_loss = running / max(1, len(train_loader))

        result.history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_macro_f1": val_metrics.macro_f1,
            "val_f1": val_metrics.f1,
            "val_pr_auc": val_metrics.pr_auc,
        })
        log.info("[%s seed=%d] epoch %d  loss %.4f  %s",
                 arm, seed, epoch, train_loss, val_metrics.summary())

        if val_metrics.macro_f1 > best_score:
            best_score = val_metrics.macro_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            result.best_epoch = epoch
            result.val = val_metrics
            epochs_without_gain = 0
        else:
            epochs_without_gain += 1
            if epochs_without_gain >= patience:
                log.info("[%s seed=%d] no gain in %d epochs — stopping at %d",
                         arm, seed, patience, epoch)
                result.epochs_run = epoch
                break
        result.epochs_run = epoch

    if best_state is not None:
        model.load_state_dict(best_state)

    # The decision threshold is chosen on VALIDATION and then frozen. Tuning it
    # on test is fitting to the test set with extra steps.
    if result.val is not None:
        y_true, y_score = predict(model, val_loader, device, arm)
        result.decision_threshold, _ = best_threshold(y_true, y_score, metric="macro_f1")
        result.val = evaluate(y_true, y_score, threshold=result.decision_threshold)

    result.train_seconds = time.perf_counter() - started

    if checkpoint_dir:
        ckpt = Path(checkpoint_dir) / f"{arm}_seed{seed}.pt"
        ckpt.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "arm": arm, "seed": seed,
                    "threshold": result.decision_threshold}, ckpt)
        log.info("checkpoint -> %s", ckpt)

    return result


def evaluate_arm(model, loader, arm: str, threshold: float = 0.5,
                 device: str | None = None) -> tuple[Metrics, np.ndarray, np.ndarray]:
    """Score a trained arm on a loader at a fixed threshold.

    Returns the metrics plus the raw labels and scores, so the PR curve, the
    error analysis and the Gate 4 image-only subset can all be computed from one
    forward pass instead of three.
    """
    torch = require_torch()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    y_true, y_score = predict(model, loader, device, arm)
    return evaluate(y_true, y_score, threshold=threshold), y_true, y_score
