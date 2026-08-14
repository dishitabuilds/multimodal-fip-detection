"""Text and image encoders, wrapped behind one small interface.

Each wrapper exposes the same contract — `forward(...) -> (sequence, pooled)` —
so the fusion module and the three ablation arms do not care whether the text
came from MuRIL or IndicBERT, or the image from ViT or CLIP. Swapping a
backbone is then a config change, which is what makes "report both, pick the
better one" (docs/CHECKLIST.md Phase 2) a one-line experiment rather than a
rewrite.

torch and transformers are imported lazily. The rest of the package — the
budget, the metrics, the example builder — must stay importable on a machine
with neither installed, because that is where the data work happens.
"""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)

_INSTALL_HINT = (
    'model code needs the optional extras: pip install -e ".[model]"'
)


def require_torch():
    """Import torch, or fail with the command that fixes it."""
    try:
        import torch  # noqa: F401
        import torch.nn as nn  # noqa: F401
    except ImportError as e:
        raise ImportError(f"{e}. {_INSTALL_HINT}") from e
    return torch


def require_transformers():
    try:
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError(f"{e}. {_INSTALL_HINT}") from e
    return transformers


def _nn():
    require_torch()
    import torch.nn as nn

    return nn


# ---------------------------------------------------------------------------


def build_text_encoder(name: str, dropout: float = 0.1, from_scratch: bool = False):
    """Wrap a HuggingFace text model (MuRIL, IndicBERT, DistilBERT, ...).

    `from_scratch` builds random weights from the config alone, downloading
    nothing. That is how the shape tests run offline and how a smoke test of
    the whole training loop stays cheap.
    """
    nn = _nn()
    transformers = require_transformers()

    class TextEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            cfg = transformers.AutoConfig.from_pretrained(name)
            if from_scratch:
                self.backbone = transformers.AutoModel.from_config(cfg)
            else:
                self.backbone = transformers.AutoModel.from_pretrained(name)
            self.dropout = nn.Dropout(dropout)
            self.hidden_size = int(cfg.hidden_size)
            self.name = name

        def forward(self, input_ids, attention_mask=None, **kw):
            out = self.backbone(input_ids=input_ids, attention_mask=attention_mask, **kw)
            seq = out.last_hidden_state                    # (B, T, H)
            # Mean-pool over real tokens. CLS is an option, but it is only
            # meaningful for models pretrained with a sentence-level objective,
            # and DistilBERT is not. Masked mean works for every backbone here,
            # so the arms stay comparable.
            if attention_mask is None:
                pooled = seq.mean(dim=1)
            else:
                m = attention_mask.unsqueeze(-1).to(seq.dtype)
                pooled = (seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            return seq, self.dropout(pooled)

    return TextEncoder()


def build_image_encoder(name: str, dropout: float = 0.1, from_scratch: bool = False):
    """Wrap a HuggingFace vision model (ViT, CLIP vision tower, ...)."""
    nn = _nn()
    transformers = require_transformers()

    class ImageEncoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            cfg = transformers.AutoConfig.from_pretrained(name)
            # CLIP checkpoints nest the vision tower's config one level down.
            if hasattr(cfg, "vision_config"):
                cfg = cfg.vision_config
                loader = transformers.CLIPVisionModel
            else:
                loader = transformers.AutoModel
            self.backbone = (loader.from_config(cfg) if from_scratch
                             else loader.from_pretrained(name))
            self.dropout = nn.Dropout(dropout)
            self.hidden_size = int(cfg.hidden_size)
            self.name = name

        def forward(self, pixel_values, **kw):
            out = self.backbone(pixel_values=pixel_values, **kw)
            seq = out.last_hidden_state                    # (B, P+1, H)
            # Patch tokens carry the layout of a doctored chart; the pooled
            # vector alone loses it. Both are returned, and the fused arm
            # attends over the patches.
            pooled = getattr(out, "pooler_output", None)
            if pooled is None:
                pooled = seq[:, 0]
            return seq, self.dropout(pooled)

    return ImageEncoder()


def freeze(module, unfreeze_last_n: int = 0) -> Any:
    """Freeze a backbone, optionally leaving its last N layers trainable.

    Useful when the dataset is small — which, per Gate 1, it may well be. A
    fully fine-tuned 238 M-parameter encoder on a few hundred examples memorises
    rather than learns.
    """
    for p in module.parameters():
        p.requires_grad = False
    if unfreeze_last_n <= 0:
        return module

    layers = None
    for attr in ("encoder.layer", "encoder.layers", "transformer.layer"):
        obj = module
        for part in attr.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            layers = obj
            break

    if layers is None:
        log.warning("could not locate transformer layers on %s — left fully frozen",
                    type(module).__name__)
        return module

    for layer in list(layers)[-unfreeze_last_n:]:
        for p in layer.parameters():
            p.requires_grad = True
    return module
