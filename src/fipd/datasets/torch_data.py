"""torch Dataset and collate function over `Example`s.

Kept deliberately thin. Everything that decides *what* an example is — the
split, the label, the text assembly, which image counts as the creative — lives
in `examples.py` and is testable without torch. This file only turns those into
tensors.

One behaviour worth knowing about: an example whose image is missing or corrupt
yields a zero image rather than raising. Collection is resumable and partial by
design, so a handful of broken files is normal, and killing a training run at
epoch 3 over one truncated JPEG is worse than feeding it a blank. Every such
substitution is counted and reported.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

from ..models.encoders import require_torch, require_transformers
from .examples import Example

log = logging.getLogger(__name__)

#: ImageNet statistics — what ViT and CLIP checkpoints were normalised with.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_dataset(
    examples: Sequence[Example],
    tokenizer_name: str,
    image_root: str | Path = "data",
    image_size: int = 224,
    max_tokens: int = 128,
    needs_image: bool = True,
    train: bool = False,
):
    """A torch Dataset over examples.

    `train` enables light augmentation. It is light on purpose: aggressive crops
    and colour jitter destroy exactly the evidence this project is looking for —
    a truncated y-axis, a pasted SEBI logo, the digits in a fake P&L screenshot.
    """
    torch = require_torch()
    require_transformers()
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer

    try:
        from PIL import Image
        from torchvision import transforms
    except ImportError as e:  # pragma: no cover - dependency shape, not logic
        raise ImportError(f'{e}. pip install -e ".[model]"') from e

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    image_root = Path(image_root)

    if train:
        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.0),  # OFF: text in the image
            transforms.ColorJitter(brightness=0.1, contrast=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    else:
        tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])

    class ExampleDataset(Dataset):
        def __init__(self) -> None:
            self.examples = list(examples)
            self.n_missing_images = 0

        def __len__(self) -> int:
            return len(self.examples)

        def _image(self, ex: Example):
            if not needs_image:
                return None
            path = image_root / ex.image_path if ex.image_path else None
            if path is None or not path.exists():
                self.n_missing_images += 1
                return torch.zeros(3, image_size, image_size)
            try:
                with Image.open(path) as im:
                    return tf(im.convert("RGB"))
            except Exception as e:  # noqa: BLE001 - one bad file must not stop a run
                log.warning("unreadable image %s (%s) — using a blank", path, e)
                self.n_missing_images += 1
                return torch.zeros(3, image_size, image_size)

        def __getitem__(self, idx: int) -> dict:
            ex = self.examples[idx]
            enc = tokenizer(
                ex.text,
                truncation=True,
                max_length=max_tokens,
                padding="max_length",
                return_tensors="pt",
            )
            item = {
                "input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": torch.tensor(ex.label, dtype=torch.long),
                "uid": ex.uid,
            }
            img = self._image(ex)
            if img is not None:
                item["pixel_values"] = img
            return item

    ds = ExampleDataset()
    log.info("dataset: %d examples, tokenizer %s, images %s",
             len(ds), tokenizer_name, "on" if needs_image else "off")
    return ds


def collate(batch: list[dict]) -> dict:
    """Stack a batch, keeping `uid` as a list so failures stay traceable."""
    torch = require_torch()
    out: dict = {}
    for key in batch[0]:
        if key == "uid":
            out[key] = [b[key] for b in batch]
        else:
            out[key] = torch.stack([b[key] for b in batch])
    return out


def build_loader(dataset, batch_size: int = 16, shuffle: bool = False,
                 num_workers: int = 0, seed: int | None = None):
    """DataLoader with a seeded generator, so a shuffle is reproducible."""
    torch = require_torch()
    from torch.utils.data import DataLoader

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        generator=generator,
        drop_last=False,
    )
