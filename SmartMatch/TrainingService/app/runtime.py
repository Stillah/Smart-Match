from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

SERVICE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_ROOT.parents[2]
TRAINING_ROOT = PROJECT_ROOT / "SmartMatch" / "TrainingContainer"
RUNTIME_ROOT = Path(os.getenv("SMARTMATCH_RUNTIME_DIR", PROJECT_ROOT / ".runtime")).resolve()
TRAINING_JOBS_DIR = RUNTIME_ROOT / "training_jobs"


def ensure_runtime_directories() -> None:
    TRAINING_JOBS_DIR.mkdir(parents=True, exist_ok=True)


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def create_training_job_dir() -> tuple[str, Path]:
    ensure_runtime_directories()
    job_id = new_run_id("train")
    job_dir = TRAINING_JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    return job_id, job_dir


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def ensure_safe_relative_path(base_dir: Path, relative_path: str) -> Path:
    candidate = (base_dir / relative_path).resolve()
    base_resolved = base_dir.resolve()
    if base_resolved == candidate or base_resolved in candidate.parents:
        return candidate
    raise ValueError("Path escapes the runtime directory.")
