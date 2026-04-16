from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    CONFIG = json.load(handle)
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _candidate_model_paths(raw_path: str) -> list[Path]:
    raw_value = str(raw_path)
    path = Path(raw_value)
    candidates = [path]
    if raw_value.startswith("/"):
        candidates.append((PROJECT_ROOT / raw_value.lstrip("/")).resolve())
    elif not path.is_absolute():
        candidates.append((Path(__file__).resolve().parent / raw_value).resolve())

    resolved_candidates = []
    seen = set()
    for candidate in candidates:
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        resolved_candidates.append(candidate)
        seen.add(candidate_str)
    return resolved_candidates


def _model_paths_from_env() -> dict[str, str]:
    env_value = os.getenv("SMARTMATCH_TROCR_MODELS")
    if env_value:
        model_paths = {}
        for raw_path in env_value.split(os.pathsep):
            for path in _candidate_model_paths(raw_path):
                if not path.exists():
                    continue
                model_name = path.name.lower()
                model_paths[model_name] = str(path)
                break
        return model_paths

    configured = CONFIG.get("trocr_model_paths", [])
    if isinstance(configured, dict):
        model_paths = {}
        for key, value in configured.items():
            for path in _candidate_model_paths(str(value)):
                if path.exists():
                    model_paths[str(key).lower()] = str(path)
                    break
        return model_paths

    model_paths = {}
    for raw_path in configured:
        for path in _candidate_model_paths(str(raw_path)):
            if path.exists():
                model_paths[path.name.lower()] = str(path)
                break
    return model_paths


def _configured_default_model() -> str:
    return str(
        os.getenv("SMARTMATCH_DEFAULT_HANDWRITTEN_MODEL", CONFIG.get("default_handwritten_model", "best"))
    ).strip().lower()


def _base_model_specs_from_config() -> dict[str, dict[str, str]]:
    model_specs: dict[str, dict[str, str]] = {}
    configured = CONFIG.get("trocr_base_models", {})
    env_configured = {
        "kazars_base": {
            "model_id": os.getenv("SMARTMATCH_TROCR_BASE_MODEL_KAZARS"),
            "processor_id": os.getenv("SMARTMATCH_TROCR_BASE_PROCESSOR_KAZARS"),
        },
        "cyrillic_base": {
            "model_id": os.getenv("SMARTMATCH_TROCR_BASE_MODEL_CYRILLIC"),
            "processor_id": os.getenv("SMARTMATCH_TROCR_BASE_PROCESSOR_CYRILLIC"),
        },
    }
    for model_name, options in (configured.items() if isinstance(configured, dict) else []):
        if not isinstance(options, dict):
            continue
        model_id = str(options.get("model_id", "")).strip()
        if not model_id:
            continue
        processor_id = str(options.get("processor_id") or model_id).strip()
        normalized_name = str(model_name).lower()
        model_specs[normalized_name] = {
            "model_name": normalized_name,
            "model_path": model_id,
            "processor_path": processor_id,
            "source": "huggingface",
        }
    for model_name, options in env_configured.items():
        model_id = str(options.get("model_id") or "").strip()
        if not model_id:
            continue
        processor_id = str(options.get("processor_id") or model_id).strip()
        model_specs[model_name] = {
            "model_name": model_name,
            "model_path": model_id,
            "processor_path": processor_id,
            "source": "huggingface",
        }
    return model_specs


def _configured_model_specs() -> dict[str, dict[str, str]]:
    model_specs = _base_model_specs_from_config()
    for model_name, model_path in _model_paths_from_env().items():
        model_specs[model_name] = {
            "model_name": model_name,
            "model_path": model_path,
            "processor_path": model_path,
            "source": "local",
        }
    return model_specs


def _preprocess(image: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = ImageEnhance.Contrast(gray).enhance(1.5)
    gray = gray.resize((gray.width * 2, gray.height * 2), Image.Resampling.LANCZOS)
    return gray.convert("RGB")


class HandwrittenOCR:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_specs = _configured_model_specs()
        self.runners: dict[str, Dict[str, Any]] = {}

        if not self.model_specs:
            raise FileNotFoundError(
                "No TrOCR models were found. Configure base model IDs or set SMARTMATCH_TROCR_MODELS "
                "to valid model directories."
            )

    def available_models(self) -> list[str]:
        model_names = list(self.model_specs.keys())
        default_model = _configured_default_model()
        if default_model in model_names:
            return [default_model, *[name for name in model_names if name != default_model]]
        return model_names

    def _get_runner(self, model_name: str) -> Dict[str, Any]:
        if model_name in self.runners:
            return self.runners[model_name]

        model_spec = self.model_specs[model_name]
        processor = TrOCRProcessor.from_pretrained(model_spec["processor_path"])
        model = VisionEncoderDecoderModel.from_pretrained(model_spec["model_path"]).to(self.device)
        model.eval()
        runner = {
            "model_name": model_spec["model_name"],
            "model_path": model_spec["model_path"],
            "source": model_spec["source"],
            "processor": processor,
            "model": model,
        }
        self.runners[model_name] = runner
        return runner

    def _run_single_model(self, model_name: str, image: Image.Image) -> Dict[str, Any]:
        runner = self._get_runner(model_name)
        processor: TrOCRProcessor = runner["processor"]
        model: VisionEncoderDecoderModel = runner["model"]
        pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            outputs = model.generate(
                pixel_values,
                num_beams=int(os.getenv("SMARTMATCH_TROCR_NUM_BEAMS", CONFIG.get("trocr_num_beams", 8))),
                early_stopping=True,
                max_new_tokens=int(
                    os.getenv("SMARTMATCH_TROCR_MAX_NEW_TOKENS", CONFIG.get("trocr_max_new_tokens", 128))
                ),
                length_penalty=float(
                    os.getenv("SMARTMATCH_TROCR_LENGTH_PENALTY", CONFIG.get("trocr_length_penalty", 0.8))
                ),
                no_repeat_ngram_size=int(
                    os.getenv(
                        "SMARTMATCH_TROCR_NO_REPEAT_NGRAM_SIZE",
                        CONFIG.get("trocr_no_repeat_ngram_size", 3),
                    )
                ),
                return_dict_in_generate=True,
                output_scores=True,
            )

        text = processor.batch_decode(outputs.sequences, skip_special_tokens=True)[0].strip()
        score = None
        if outputs.sequences_scores is not None:
            score = float(outputs.sequences_scores[0].item())
        return {
            "model_name": runner["model_name"],
            "model_path": runner["model_path"],
            "source": runner["source"],
            "text": text,
            "score": score,
        }

    def recognize(self, image_path: str, model_name: str | None = None) -> Dict[str, Any]:
        requested_model = (model_name or _configured_default_model()).lower()
        image = Image.open(image_path).convert("RGB")
        image = _preprocess(image)

        if requested_model != "best" and requested_model not in self.model_specs:
            raise ValueError(
                f"Unknown handwritten OCR model '{model_name}'. "
                f"Available: {', '.join(self.available_models())}"
            )

        candidate_names = self.available_models() if requested_model == "best" else [requested_model]

        candidates = []
        best_candidate = None
        best_score = float("-inf")
        for candidate_name in candidate_names:
            candidate = self._run_single_model(candidate_name, image)
            candidates.append(candidate)
            candidate_score = candidate["score"] if candidate["score"] is not None else float("-inf")
            if candidate["text"] and candidate_score >= best_score:
                best_score = candidate_score
                best_candidate = candidate

        if best_candidate is None:
            best_candidate = candidates[0]

        return {
            "status": "done",
            "engine": "trocr",
            "device": str(self.device),
            "selected_model": requested_model,
            "text": best_candidate.get("text", ""),
            "confidence": best_candidate.get("score"),
            "model_name": best_candidate.get("model_name"),
            "model_path": best_candidate.get("model_path"),
            "candidates": candidates,
        }
