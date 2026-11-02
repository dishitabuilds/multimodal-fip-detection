"""Local random HF snapshots for an opt-in plumbing smoke, with no downloads.

The UTF-8 byte tokenizer represents English and Hindi without an unknown-token
vocabulary gap. These encoders have NO pretrained language or vision knowledge;
successful training checks the plumbing, not accuracy or teacher retention.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from .encoders import require_torch, require_transformers

ORIGIN = "random_init_untrained_smoke_baseline"
FORMAT_VERSION = 1


def prepare_tiny_assets(output_dir: str | Path, seed: int = 42) -> dict:
    """Write local HF text/image snapshots, refusing to overwrite existing ones.

    Both encoders use width 128, two blocks and FFN width 256. The vocabulary
    contains every UTF-8 byte plus five special tokens. ViT retains 224px inputs
    and 16px patches. These dimensions are our smoke design, not metadata for
    any published pretrained checkpoint.
    """
    torch = require_torch()
    transformers = require_transformers()
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite tiny assets: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    special = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
    vocabulary = {token: index for index, token in enumerate(special)}
    vocabulary.update({token: index + len(special) for index, token in
                       enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))})
    raw_tokenizer = Tokenizer(models.BPE(vocab=vocabulary, merges=[], unk_token="[UNK]"))
    raw_tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    raw_tokenizer.decoder = decoders.ByteLevel()
    raw_tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]", pair="[CLS] $A [SEP] $B [SEP]",
        special_tokens=[("[CLS]", vocabulary["[CLS]"]), ("[SEP]", vocabulary["[SEP]"])],
    )
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer, model_max_length=256,
        pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
        sep_token="[SEP]", mask_token="[MASK]",
    )
    text_config = transformers.DistilBertConfig(
        vocab_size=len(vocabulary), dim=128, hidden_dim=256, n_layers=2,
        n_heads=4, max_position_embeddings=256, dropout=0.1, attention_dropout=0.1,
        pad_token_id=vocabulary["[PAD]"],
        fipd_weight_origin=ORIGIN, fipd_initialization_seed=seed,
    )
    image_config = transformers.ViTConfig(
        image_size=224, patch_size=16, num_channels=3, hidden_size=128,
        intermediate_size=256, num_hidden_layers=2, num_attention_heads=4,
        qkv_bias=True, fipd_weight_origin=ORIGIN, fipd_initialization_seed=seed,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        text = transformers.DistilBertModel(text_config).float()
        image = transformers.ViTModel(image_config).float()
    manifest = {
        "format_version": FORMAT_VERSION, "weight_origin": ORIGIN, "seed": seed,
        "pretrained": False, "tokenizer": "UTF-8 bytes, no learned BPE merges",
        "vocab_size": len(vocabulary), "max_text_tokens": 256, "image_size": 224,
        "text_parameters": sum(parameter.numel() for parameter in text.parameters()),
        "image_parameters": sum(parameter.numel() for parameter in image.parameters()),
        "torch": torch.__version__, "transformers": transformers.__version__,
    }
    # Publish the directory only after both snapshots and the tokenizer exist.
    with tempfile.TemporaryDirectory(prefix=".tiny-smoke-", dir=output_dir.parent) as temp:
        staging = Path(temp) / "assets"
        text.save_pretrained(staging / "text")
        tokenizer.save_pretrained(staging / "text")
        image.save_pretrained(staging / "image")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staging.rename(output_dir)
    return manifest
