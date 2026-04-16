"""Fine-tune cyrillic-trocr/trocr-handwritten-cyrillic on paired image-text data."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrOCRProcessor,
    VisionEncoderDecoderModel,
    default_data_collator,
)

from env_config import default_output_root, joined_data_dir, trocr_base_model, trocr_processor_id

MODEL_ID = trocr_base_model("cyrillic")
TOKENIZER_ID = trocr_processor_id("cyrillic")


def train(
    train_pairs: list[tuple[Path, Path]],
    output_dir: str | Path,
    epochs: int = 3,
    batch_size: int = 8,
    lr: float = 5e-5,
    max_target_length: int = 128,
    warmup_steps: int = 100,
    logging_steps: int = 50,
    save_total_limit: int = 2,
) -> None:
    from dataset import OCRDataset

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = TrOCRProcessor.from_pretrained(TOKENIZER_ID)
    model = VisionEncoderDecoderModel.from_pretrained(MODEL_ID)

    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.max_new_tokens = max_target_length

    image_paths, text_paths = zip(*train_pairs)
    train_dataset = OCRDataset(
        list(image_paths),
        list(text_paths),
        processor,
        max_target_length=max_target_length,
    )

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        save_strategy="epoch",
        save_total_limit=save_total_limit,
        logging_steps=logging_steps,
        fp16=torch.cuda.is_available(),
        predict_with_generate=True,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=default_data_collator,
    )
    trainer.train()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    from dataset import load_data_pairs, split_pairs

    data_dir = joined_data_dir()
    pairs = load_data_pairs(data_dir, num_samples=200)
    train_pairs, _ = split_pairs(pairs)
    train(
        train_pairs=train_pairs,
        output_dir=default_output_root() / "cyrillic",
        epochs=1,
        batch_size=4,
    )
