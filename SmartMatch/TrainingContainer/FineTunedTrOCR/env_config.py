from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> Path:
    raw_value = os.getenv(name)
    if raw_value:
        return Path(raw_value).resolve()
    return default.resolve()


def joined_data_dir() -> Path:
    return _env_path("SMARTMATCH_JOINED_DATA_DIR", PROJECT_ROOT / "OCR" / "joined_data")


def default_output_root() -> Path:
    return (Path(__file__).parent / "outputs").resolve()


def trocr_base_model(model_name: str) -> str:
    normalized = model_name.strip().lower()
    mapping = {
        "kazars": "SMARTMATCH_TROCR_BASE_MODEL_KAZARS",
        "cyrillic": "SMARTMATCH_TROCR_BASE_MODEL_CYRILLIC",
    }
    defaults = {
        "kazars": "kazars24/trocr-base-handwritten-ru",
        "cyrillic": "cyrillic-trocr/trocr-handwritten-cyrillic",
    }
    env_name = mapping[normalized]
    return os.getenv(env_name, defaults[normalized]).strip()


def trocr_processor_id(model_name: str) -> str:
    normalized = model_name.strip().lower()
    mapping = {
        "kazars": "SMARTMATCH_TROCR_BASE_PROCESSOR_KAZARS",
        "cyrillic": "SMARTMATCH_TROCR_BASE_PROCESSOR_CYRILLIC",
    }
    defaults = {
        "kazars": trocr_base_model("kazars"),
        "cyrillic": trocr_base_model("kazars"),
    }
    env_name = mapping[normalized]
    return os.getenv(env_name, defaults[normalized]).strip()
