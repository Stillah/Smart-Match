"""Shared dataset utilities for TrOCR finetuning."""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class OCRDataset(Dataset):
    def __init__(
        self,
        image_paths: list[Path],
        text_paths: list[Path],
        processor,
        max_target_length: int = 128,
    ) -> None:
        self.image_paths = image_paths
        self.text_paths = text_paths
        self.processor = processor
        self.max_target_length = max_target_length

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        with open(self.text_paths[idx], encoding="utf-8") as handle:
            text = handle.read().strip()

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.squeeze(0)
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_target_length,
            truncation=True,
            return_tensors="pt",
        ).input_ids.squeeze(0)
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        return {"pixel_values": pixel_values, "labels": labels}


def load_data_pairs(
    data_dir: str | Path,
    num_samples: int | None = None,
    seed: int = 42,
) -> list[tuple[Path, Path]]:
    data_dir = Path(data_dir)
    images_dir = data_dir / "images"
    texts_dir = data_dir / "texts"

    pairs: list[tuple[Path, Path]] = []
    for img_path in sorted(images_dir.glob("*.jpg")):
        txt_path = texts_dir / f"{img_path.stem}.txt"
        if txt_path.exists():
            pairs.append((img_path, txt_path))

    if num_samples is not None and num_samples < len(pairs):
        rng = random.Random(seed)
        pairs = pairs.copy()
        rng.shuffle(pairs)
        pairs = pairs[:num_samples]
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float = 0.9,
    seed: int = 42,
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    shuffled = pairs.copy()
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    return shuffled[:split_idx], shuffled[split_idx:]
