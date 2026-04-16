from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from runtime import PROJECT_ROOT, TRAINING_JOBS_DIR, TRAINING_ROOT, create_training_job_dir, load_json, save_json

_RUNNING_JOBS: dict[str, subprocess.Popen[Any]] = {}


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return default
    return int(raw_value)


def _env_float(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return default
    return float(raw_value)


def _env_optional_int(name: str, default: int | None = None) -> int | None:
    raw_value = os.getenv(name)
    if raw_value in {None, ""}:
        return default
    return int(raw_value)


def _training_data_root() -> Path:
    return Path(os.getenv("SMARTMATCH_TRAINING_DATA_CONTAINER_DIR", PROJECT_ROOT / "data")).resolve()


def _joined_data_candidates() -> list[Path]:
    candidates = []
    env_path = os.getenv("SMARTMATCH_JOINED_DATA_DIR")
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(PROJECT_ROOT / "OCR" / "joined_data")
    return candidates


def _classifier_dataset_candidates() -> list[Path]:
    candidates = []
    env_path = os.getenv("SMARTMATCH_CLASSIFIER_DATASET_DIR")
    if env_path:
        candidates.append(Path(env_path))

    data_root = _training_data_root()
    candidates.extend(
        [
            data_root / "classifier_dataset",
            data_root / "imgs",
            PROJECT_ROOT / "OCR" / "classifier_data" / "imgs",
            PROJECT_ROOT / "OCR" / "output" / "imgs",
            TRAINING_ROOT / "HandwrittenTypedClassifier" / "imgs",
        ]
    )
    return candidates


def _count_classifier_files(dataset_dir: Path) -> tuple[int, int]:
    png_count = len(list(dataset_dir.glob("*.png")))
    jpeg_count = len(list(dataset_dir.glob("*.jpg"))) + len(list(dataset_dir.glob("*.jpeg")))
    return png_count, jpeg_count


def _discover_classifier_dataset() -> tuple[Path | None, str | None]:
    for candidate in _classifier_dataset_candidates():
        if not candidate.exists() or not candidate.is_dir():
            continue
        png_count, jpeg_count = _count_classifier_files(candidate)
        if png_count > 0 and jpeg_count > 0:
            return candidate.resolve(), None
    return None, "Classifier dataset with both PNG and JPEG classes was not found."


def _discover_joined_data() -> tuple[Path | None, str | None]:
    for candidate in _joined_data_candidates():
        candidate = candidate.resolve()
        images_dir = candidate / "images"
        texts_dir = candidate / "texts"
        if not images_dir.exists() or not texts_dir.exists():
            continue
        if not any(images_dir.glob("*.jpg")):
            continue
        if not any(texts_dir.glob("*.txt")):
            continue
        return candidate, None
    return None, "joined_data/images and joined_data/texts are required."


def _trocr_defaults() -> dict[str, Any]:
    return {
        "epochs": _env_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_EPOCHS", 10),
        "batch_size": _env_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_BATCH_SIZE", 8),
        "learning_rate": _env_float("SMARTMATCH_TRAINING_DEFAULT_TROCR_LEARNING_RATE", 3.0e-5),
        "num_samples": _env_optional_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_NUM_SAMPLES", None),
        "train_split": _env_float("SMARTMATCH_TRAINING_DEFAULT_TROCR_TRAIN_SPLIT", 0.9),
        "seed": _env_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_SEED", 42),
        "max_target_length": _env_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_MAX_TARGET_LENGTH", 128),
        "warmup_steps": _env_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_WARMUP_STEPS", 100),
        "logging_steps": _env_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_LOGGING_STEPS", 50),
        "save_total_limit": _env_int("SMARTMATCH_TRAINING_DEFAULT_TROCR_SAVE_TOTAL_LIMIT", 2),
    }


def _classifier_defaults() -> dict[str, Any]:
    return {
        "batch_size": _env_int("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_BATCH_SIZE", 32),
        "num_epochs": _env_int("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_NUM_EPOCHS", 10),
        "learning_rate": _env_float("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_LEARNING_RATE", 1.0e-3),
        "patience": _env_int("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_PATIENCE", 5),
        "random_state": _env_int("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_RANDOM_STATE", 42),
        "target_height": _env_int("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_TARGET_HEIGHT", 64),
        "target_width": _env_int("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_TARGET_WIDTH", 256),
        "test_split": _env_float("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_TEST_SPLIT", 0.3),
        "val_split": _env_float("SMARTMATCH_TRAINING_DEFAULT_CLASSIFIER_VAL_SPLIT", 0.5),
    }


def list_trainable_models() -> list[dict[str, Any]]:
    trocr_defaults = _trocr_defaults()
    classifier_defaults = _classifier_defaults()
    classifier_dataset, classifier_reason = _discover_classifier_dataset()
    joined_data_dir, joined_data_reason = _discover_joined_data()

    trocr_parameters = [
        {"name": "epochs", "type": "int", "default": trocr_defaults["epochs"]},
        {"name": "batch_size", "type": "int", "default": trocr_defaults["batch_size"]},
        {"name": "learning_rate", "type": "float", "default": trocr_defaults["learning_rate"]},
        {"name": "num_samples", "type": "int", "default": trocr_defaults["num_samples"]},
        {"name": "train_split", "type": "float", "default": trocr_defaults["train_split"]},
        {"name": "seed", "type": "int", "default": trocr_defaults["seed"]},
        {"name": "max_target_length", "type": "int", "default": trocr_defaults["max_target_length"]},
        {"name": "warmup_steps", "type": "int", "default": trocr_defaults["warmup_steps"]},
        {"name": "logging_steps", "type": "int", "default": trocr_defaults["logging_steps"]},
        {"name": "save_total_limit", "type": "int", "default": trocr_defaults["save_total_limit"]},
    ]
    classifier_parameters = [
        {"name": "batch_size", "type": "int", "default": classifier_defaults["batch_size"]},
        {"name": "num_epochs", "type": "int", "default": classifier_defaults["num_epochs"]},
        {"name": "learning_rate", "type": "float", "default": classifier_defaults["learning_rate"]},
        {"name": "patience", "type": "int", "default": classifier_defaults["patience"]},
        {"name": "random_state", "type": "int", "default": classifier_defaults["random_state"]},
        {"name": "target_height", "type": "int", "default": classifier_defaults["target_height"]},
        {"name": "target_width", "type": "int", "default": classifier_defaults["target_width"]},
        {"name": "test_split", "type": "float", "default": classifier_defaults["test_split"]},
        {"name": "val_split", "type": "float", "default": classifier_defaults["val_split"]},
    ]

    return [
        {
            "key": "trocr_kazars",
            "title": "TrOCR Kazars",
            "available": joined_data_dir is not None,
            "reason": joined_data_reason,
            "dataset_path": str(joined_data_dir) if joined_data_dir else None,
            "parameters": trocr_parameters,
        },
        {
            "key": "trocr_cyrillic",
            "title": "TrOCR Cyrillic",
            "available": joined_data_dir is not None,
            "reason": joined_data_reason,
            "dataset_path": str(joined_data_dir) if joined_data_dir else None,
            "parameters": trocr_parameters,
        },
        {
            "key": "handwritten_typed_classifier",
            "title": "Handwritten / Typed Classifier",
            "available": classifier_dataset is not None,
            "reason": classifier_reason,
            "dataset_path": str(classifier_dataset) if classifier_dataset else None,
            "parameters": classifier_parameters,
        },
    ]


def _write_trocr_config(job_dir: Path, model_name: str, params: dict[str, Any], dataset_dir: Path) -> tuple[Path, Path]:
    defaults = _trocr_defaults()
    output_dir = job_dir / "output"
    config_path = job_dir / "config.yaml"
    epochs = int(params.get("epochs", defaults["epochs"]))
    learning_rate = float(params.get("learning_rate", defaults["learning_rate"]))
    payload = {
        "data": {
            "dir": str(dataset_dir),
            "num_samples": int(params["num_samples"]) if params.get("num_samples") is not None else defaults["num_samples"],
            "train_split": float(params.get("train_split", defaults["train_split"])),
            "seed": int(params.get("seed", defaults["seed"])),
        },
        "output": {
            "dir": str(output_dir),
        },
        "training": {
            "batch_size": int(params.get("batch_size", defaults["batch_size"])),
            "max_target_length": int(params.get("max_target_length", defaults["max_target_length"])),
            "warmup_steps": int(params.get("warmup_steps", defaults["warmup_steps"])),
            "logging_steps": int(params.get("logging_steps", defaults["logging_steps"])),
            "save_total_limit": int(params.get("save_total_limit", defaults["save_total_limit"])),
        },
        "models": {
            "kazars": {
                "epochs": epochs if model_name == "kazars" else defaults["epochs"],
                "lr": learning_rate if model_name == "kazars" else defaults["learning_rate"],
            },
            "cyrillic": {
                "epochs": epochs if model_name == "cyrillic" else defaults["epochs"],
                "lr": learning_rate if model_name == "cyrillic" else defaults["learning_rate"],
            },
        },
    }
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path, output_dir


def _write_classifier_config(job_dir: Path, params: dict[str, Any], dataset_dir: Path) -> tuple[Path, Path]:
    defaults = _classifier_defaults()
    output_dir = job_dir / "output"
    config_path = job_dir / "config.json"
    payload = {
        "img_folder": str(dataset_dir),
        "output_dir": str(output_dir),
        "batch_size": int(params.get("batch_size", defaults["batch_size"])),
        "num_epochs": int(params.get("num_epochs", defaults["num_epochs"])),
        "learning_rate": float(params.get("learning_rate", defaults["learning_rate"])),
        "patience": int(params.get("patience", defaults["patience"])),
        "random_state": int(params.get("random_state", defaults["random_state"])),
        "target_size": [
            int(params.get("target_height", defaults["target_height"])),
            int(params.get("target_width", defaults["target_width"])),
        ],
        "test_split": float(params.get("test_split", defaults["test_split"])),
        "val_split": float(params.get("val_split", defaults["val_split"])),
    }
    save_json(config_path, payload)
    return config_path, output_dir


def _job_metadata_path(job_dir: Path) -> Path:
    return job_dir / "job.json"


def _log_tail(log_path: Path, limit: int = 50000) -> str:
    if not log_path.exists():
        return ""
    content = log_path.read_text(encoding="utf-8", errors="replace")
    return content[-limit:]


def _pid_is_alive(pid: Any) -> bool:
    if pid in {None, ""}:
        return False
    try:
        os.kill(int(pid), 0)
    except (ProcessLookupError, ValueError, TypeError):
        return False
    except PermissionError:
        return True
    return True


def _refresh_job_status(job_id: str, metadata: dict[str, Any]) -> dict[str, Any]:
    process = _RUNNING_JOBS.get(job_id)
    if process is None:
        if metadata.get("status") == "running" and not _pid_is_alive(metadata.get("pid")):
            metadata["status"] = "interrupted"
        metadata["log_tail"] = _log_tail(Path(metadata["log_path"]))
        save_json(Path(metadata["metadata_path"]), metadata)
        return metadata

    return_code = process.poll()
    if return_code is None:
        metadata["status"] = "running"
    else:
        metadata["status"] = "completed" if return_code == 0 else "failed"
        metadata["return_code"] = return_code
        _RUNNING_JOBS.pop(job_id, None)
    metadata["log_tail"] = _log_tail(Path(metadata["log_path"]))
    save_json(Path(metadata["metadata_path"]), metadata)
    return metadata


def get_training_job(job_id: str) -> dict[str, Any]:
    job_dir = TRAINING_JOBS_DIR / job_id
    metadata_path = _job_metadata_path(job_dir)
    if not metadata_path.exists():
        raise FileNotFoundError(f"Training job not found: {job_id}")
    metadata = load_json(metadata_path)
    metadata["metadata_path"] = str(metadata_path)
    return _refresh_job_status(job_id, metadata)


def list_training_jobs() -> list[dict[str, Any]]:
    jobs = []
    if not TRAINING_JOBS_DIR.exists():
        return jobs
    for metadata_path in sorted(TRAINING_JOBS_DIR.glob("*/job.json"), reverse=True):
        metadata = load_json(metadata_path)
        metadata["metadata_path"] = str(metadata_path)
        jobs.append(_refresh_job_status(metadata["job_id"], metadata))
    return jobs


def start_training_job(model_key: str, params: dict[str, Any]) -> dict[str, Any]:
    models_by_key = {model["key"]: model for model in list_trainable_models()}
    if model_key not in models_by_key:
        raise ValueError(f"Unknown training model: {model_key}")

    model_info = models_by_key[model_key]
    if not model_info["available"]:
        raise ValueError(model_info["reason"] or f"Model {model_key} is unavailable.")

    job_id, job_dir = create_training_job_dir()
    log_path = job_dir / "train.log"

    if model_key.startswith("trocr_"):
        trocr_model = "kazars" if model_key.endswith("kazars") else "cyrillic"
        dataset_dir = Path(model_info["dataset_path"])
        config_path, output_dir = _write_trocr_config(job_dir, trocr_model, params, dataset_dir)
        command = [
            sys.executable,
            str(TRAINING_ROOT / "FineTunedTrOCR" / "pipeline.py"),
            "--model",
            trocr_model,
            "--mode",
            "all",
            "--config",
            str(config_path),
        ]
    else:
        dataset_dir = Path(model_info["dataset_path"])
        config_path, output_dir = _write_classifier_config(job_dir, params, dataset_dir)
        command = [
            sys.executable,
            str(TRAINING_ROOT / "HandwrittenTypedClassifier" / "train.py"),
            "--config",
            str(config_path),
        ]

    metadata = {
        "job_id": job_id,
        "model_key": model_key,
        "model_title": model_info["title"],
        "status": "running",
        "parameters": params,
        "dataset_path": model_info["dataset_path"],
        "config_path": str(config_path),
        "output_dir": str(output_dir),
        "log_path": str(log_path),
        "command": command,
        "metadata_path": str(_job_metadata_path(job_dir)),
    }
    save_json(_job_metadata_path(job_dir), metadata)

    log_handle = open(log_path, "w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env={**os.environ, "PYTHONUNBUFFERED": "1"},
    )
    log_handle.close()

    metadata["pid"] = process.pid
    save_json(_job_metadata_path(job_dir), metadata)
    _RUNNING_JOBS[job_id] = process
    return get_training_job(job_id)
