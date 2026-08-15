"""The three ablation arms, and the cross-attention fusion that joins them.

    text_only    caption + OCR text  -> classifier
    image_only   the creative        -> classifier
    fused        both, cross-attended -> classifier

These are the deliverable, not an appendix. The claim under test is that the
deception lives in the image, so the number that matters is fused minus
text_only, with its sign and magnitude — three separate scores do not state it.

Every arm shares the same classifier head shape and the same pooled-feature
convention, so a difference between arms is a difference in what the model can
see, not in how it was wired. That is the whole point of an ablation.
"""

from __future__ import annotations

import logging

from .config import ARMS, ArchConfig
from .encoders import _nn, build_image_encoder, build_text_encoder, require_torch

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------


def build_cross_attention_fusion(dim: int, heads: int, layers: int, dropout: float):
    """Text queries the image, image queries the text, both directions kept.

    A single direction would be a choice about which modality is primary, and
    the project has no basis for that: the caption is sometimes benign filler
    around a loud image, and sometimes the reverse. Running both and
    concatenating lets the head weigh them, and costs one extra attention block.
    """
    nn = _nn()
    torch = require_torch()

    class CrossAttentionFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([
                nn.ModuleDict({
                    "t2i": nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                 batch_first=True),
                    "i2t": nn.MultiheadAttention(dim, heads, dropout=dropout,
                                                 batch_first=True),
                    "norm_t": nn.LayerNorm(dim),
                    "norm_i": nn.LayerNorm(dim),
                    "ff_t": nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(dim * 2, dim)),
                    "ff_i": nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(),
                                          nn.Dropout(dropout), nn.Linear(dim * 2, dim)),
                })
                for _ in range(max(1, layers))
            ])
            self.out_norm = nn.LayerNorm(dim * 2)
            self.out_dim = dim * 2

        def forward(self, text_seq, image_seq, text_mask=None):
            # MultiheadAttention wants True where a key must be IGNORED; the
            # tokeniser's attention_mask is the opposite convention. Getting
            # this backwards trains the model on padding and is silent.
            key_padding = None
            if text_mask is not None:
                key_padding = (text_mask == 0)

            t, i = text_seq, image_seq
            for layer in self.layers:
                t_att, _ = layer["t2i"](query=t, key=i, value=i)
                i_att, _ = layer["i2t"](query=i, key=t, value=t,
                                        key_padding_mask=key_padding)
                t = layer["norm_t"](t + t_att)
                i = layer["norm_i"](i + i_att)
                t = t + layer["ff_t"](t)
                i = i + layer["ff_i"](i)

            if text_mask is None:
                t_pool = t.mean(dim=1)
            else:
                m = text_mask.unsqueeze(-1).to(t.dtype)
                t_pool = (t * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            i_pool = i.mean(dim=1)
            return self.out_norm(torch.cat([t_pool, i_pool], dim=-1))

    return CrossAttentionFusion()


def build_concat_fusion(dim: int, dropout: float):
    """Baseline fusion: concatenate the two pooled vectors, no attention.

    Worth running. If cross-attention does not beat plain concatenation, the
    extra machinery is not earning its parameters, and that belongs in the
    report rather than being quietly omitted.
    """
    nn = _nn()
    torch = require_torch()

    class ConcatFusion(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = nn.LayerNorm(dim * 2)
            self.drop = nn.Dropout(dropout)
            self.out_dim = dim * 2

        def forward(self, text_seq, image_seq, text_mask=None):
            if text_mask is None:
                t_pool = text_seq.mean(dim=1)
            else:
                m = text_mask.unsqueeze(-1).to(text_seq.dtype)
                t_pool = (text_seq * m).sum(dim=1) / m.sum(dim=1).clamp(min=1e-9)
            i_pool = image_seq.mean(dim=1)
            return self.drop(self.norm(torch.cat([t_pool, i_pool], dim=-1)))

    return ConcatFusion()


# ---------------------------------------------------------------------------


def build_arm(arm: str, cfg: ArchConfig, from_scratch: bool = False):
    """Construct one ablation arm. `arm` is one of models.config.ARMS.

    `from_scratch` gives randomly-initialised backbones and downloads nothing —
    for shape tests and for timing the architecture before committing to a
    training run.
    """
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {list(ARMS)}")

    # Caught here rather than several layers down inside MultiheadAttention,
    # where the error names neither the config key nor the file it came from.
    if arm == "fused" and cfg.fusion == "cross_attention":
        if cfg.hidden_dim % cfg.fusion_heads:
            raise ValueError(
                f"hidden_dim ({cfg.hidden_dim}) must divide by fusion_heads "
                f"({cfg.fusion_heads}); fix them in configs/model.yaml")

    nn = _nn()

    text_enc = image_enc = fusion = None
    if arm in ("text_only", "fused"):
        text_enc = build_text_encoder(cfg.text_encoder, cfg.dropout, from_scratch)
    if arm in ("image_only", "fused"):
        image_enc = build_image_encoder(cfg.image_encoder, cfg.dropout, from_scratch)

    # Project both modalities to a shared width before they can attend to each
    # other; the backbones rarely agree on hidden size (MuRIL 768, ViT-S 384).
    dim = cfg.hidden_dim
    text_proj = nn.Linear(text_enc.hidden_size, dim) if text_enc else None
    image_proj = nn.Linear(image_enc.hidden_size, dim) if image_enc else None

    if arm == "fused":
        fusion = (build_cross_attention_fusion(dim, cfg.fusion_heads,
                                               cfg.fusion_layers, cfg.dropout)
                  if cfg.fusion == "cross_attention"
                  else build_concat_fusion(dim, cfg.dropout))
        head_in = fusion.out_dim
    else:
        head_in = dim

    class Arm(nn.Module):
        """One ablation arm: encoders -> (fusion) -> classifier."""

        def __init__(self) -> None:
            super().__init__()
            self.arm = arm
            self.text_encoder = text_enc
            self.image_encoder = image_enc
            self.text_proj = text_proj
            self.image_proj = image_proj
            self.fusion = fusion
            self.classifier = nn.Sequential(
                nn.Dropout(cfg.dropout),
                nn.Linear(head_in, dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(dim, cfg.num_labels),
            )

        def forward(self, input_ids=None, attention_mask=None, pixel_values=None):
            if self.arm == "text_only":
                if input_ids is None:
                    raise ValueError("text_only arm requires input_ids")
                seq, pooled = self.text_encoder(input_ids, attention_mask)
                features = self.text_proj(pooled)

            elif self.arm == "image_only":
                if pixel_values is None:
                    raise ValueError("image_only arm requires pixel_values")
                seq, pooled = self.image_encoder(pixel_values)
                features = self.image_proj(pooled)

            else:
                if input_ids is None or pixel_values is None:
                    raise ValueError("fused arm requires both input_ids and pixel_values")
                t_seq, _ = self.text_encoder(input_ids, attention_mask)
                i_seq, _ = self.image_encoder(pixel_values)
                features = self.fusion(self.text_proj(t_seq),
                                       self.image_proj(i_seq),
                                       text_mask=attention_mask)

            return self.classifier(features)

    model = Arm()
    n_params = sum(p.numel() for p in model.parameters())
    log.info("built %s arm: %.1f M parameters", arm, n_params / 1e6)
    return model
