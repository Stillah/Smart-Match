"""Evaluate a fine-tuned (or base) kazars24/trocr-base-handwritten-ru model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

from env_config import joined_data_dir, trocr_base_model, trocr_processor_id

BASE_MODEL_ID = trocr_base_model("kazars")


def _edit_distance(a: str, b: str) -> int:
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def _cer(predictions: list[str], references: list[str]) -> float:
    total_errors = sum(_edit_distance(pred, ref) for pred, ref in zip(predictions, references))
    total_chars = sum(len(ref) for ref in references)
    return total_errors / total_chars if total_chars > 0 else 0.0


def evaluate(
    eval_pairs: list[tuple[Path, Path]],
    model_path: str | Path,
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> dict:
    model_path = str(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor_path = model_path if Path(model_path).is_dir() else trocr_processor_id("kazars")
    processor = TrOCRProcessor.from_pretrained(processor_path)
    model = VisionEncoderDecoderModel.from_pretrained(model_path)
    model.to(device)
    model.eval()

    predictions: list[str] = []
    references: list[str] = []

    for batch_start in range(0, len(eval_pairs), batch_size):
        batch = eval_pairs[batch_start : batch_start + batch_size]
        images = [Image.open(img).convert("RGB") for img, _ in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)

        with torch.no_grad():
            generated_ids = model.generate(pixel_values, max_new_tokens=max_new_tokens)

        predictions.extend(processor.batch_decode(generated_ids, skip_special_tokens=True))
        for _, txt_path in batch:
            with open(txt_path, encoding="utf-8") as handle:
                references.append(handle.read().strip())

    cer = _cer(predictions, references)
    return {
        "cer": round(cer, 6),
        "num_samples": len(eval_pairs),
        "model_path": model_path,
    }


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import load_data_pairs, split_pairs

    data_dir = joined_data_dir()
    pairs = load_data_pairs(data_dir, num_samples=100)
    _, eval_pairs = split_pairs(pairs)
    print(evaluate(eval_pairs=eval_pairs, model_path=BASE_MODEL_ID))
