"""TrOCR fine-tuning pipeline."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
DEFAULT_CONFIG = Path(__file__).parent / "config.yaml"


def _load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        sys.exit(f"[ERROR] Config file not found: {config_path}")
    with open(config_path, encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)

    data_dir = Path(cfg["data"]["dir"])
    if not data_dir.is_absolute():
        data_dir = (config_path.parent / data_dir).resolve()
    cfg["data"]["dir"] = str(data_dir)

    out_dir = Path(cfg["output"]["dir"])
    if not out_dir.is_absolute():
        out_dir = (config_path.parent / out_dir).resolve()
    cfg["output"]["dir"] = str(out_dir)

    cfg["data"]["num_samples"] = (
        int(cfg["data"]["num_samples"]) if cfg["data"]["num_samples"] is not None else None
    )
    cfg["data"]["train_split"] = float(cfg["data"]["train_split"])
    cfg["data"]["seed"] = int(cfg["data"]["seed"])
    cfg["training"]["batch_size"] = int(cfg["training"]["batch_size"])
    cfg["training"]["max_target_length"] = int(cfg["training"]["max_target_length"])
    cfg["training"]["warmup_steps"] = int(cfg["training"]["warmup_steps"])
    cfg["training"]["logging_steps"] = int(cfg["training"]["logging_steps"])
    cfg["training"]["save_total_limit"] = int(cfg["training"]["save_total_limit"])
    for model_name in ("kazars", "cyrillic"):
        cfg["models"][model_name]["epochs"] = int(cfg["models"][model_name]["epochs"])
        cfg["models"][model_name]["lr"] = float(cfg["models"][model_name]["lr"])
    return cfg


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TrOCR fine-tuning pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=["kazars", "cyrillic"], required=True)
    parser.add_argument("--mode", choices=["train", "eval", "all"], default="all")
    parser.add_argument("--eval-base-model", action="store_true")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    cfg = _load_config(args.config)

    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    model_cfg = cfg["models"][args.model]
    output_root = cfg["output"]["dir"]

    from dataset import load_data_pairs, split_pairs

    pairs = load_data_pairs(
        data_cfg["dir"],
        num_samples=data_cfg["num_samples"],
        seed=data_cfg["seed"],
    )
    if not pairs:
        sys.exit(
            f"[ERROR] No matched image-text pairs found in '{data_cfg['dir']}'. "
            "Make sure images/ and texts/ sub-directories exist."
        )

    train_pairs, eval_pairs = split_pairs(
        pairs,
        train_ratio=data_cfg["train_split"],
        seed=data_cfg["seed"],
    )

    output_dir = Path(output_root) / args.model
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "kazars":
        from eval_kazars import BASE_MODEL_ID, evaluate
        from train_kazars import train
    else:
        from eval_cyrillic import BASE_MODEL_ID, evaluate
        from train_cyrillic import train

    if args.mode in ("train", "all"):
        train(
            train_pairs=train_pairs,
            output_dir=str(output_dir),
            epochs=model_cfg["epochs"],
            batch_size=training_cfg["batch_size"],
            lr=model_cfg["lr"],
            max_target_length=training_cfg["max_target_length"],
            warmup_steps=training_cfg["warmup_steps"],
            logging_steps=training_cfg["logging_steps"],
            save_total_limit=training_cfg["save_total_limit"],
        )

    if args.mode in ("eval", "all"):
        model_path = BASE_MODEL_ID if args.eval_base_model else str(output_dir)
        print(
            evaluate(
                eval_pairs=eval_pairs,
                model_path=model_path,
                batch_size=training_cfg["batch_size"],
                max_new_tokens=training_cfg["max_target_length"],
            )
        )


if __name__ == "__main__":
    main()
