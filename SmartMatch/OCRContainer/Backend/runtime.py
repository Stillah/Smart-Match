from __future__ import annotations

import json
import mimetypes
import os
import uuid
from pathlib import Path
from typing import Any, Iterable

CONTAINER_ROOT = Path(__file__).resolve().parent.parent
BACKEND_ROOT = Path(__file__).resolve().parent
FRONTEND_ROOT = BACKEND_ROOT / "frontend"
PROJECT_ROOT = CONTAINER_ROOT.parents[1]
RUNTIME_ROOT = Path(os.getenv("SMARTMATCH_RUNTIME_DIR", PROJECT_ROOT / ".runtime")).resolve()
LOG_DIR = RUNTIME_ROOT / "logs"
OCR_RUNS_DIR = RUNTIME_ROOT / "ocr_runs"
TRAINING_JOBS_DIR = RUNTIME_ROOT / "training_jobs"


def ensure_runtime_directories() -> None:
    for directory in (FRONTEND_ROOT, LOG_DIR, OCR_RUNS_DIR, TRAINING_JOBS_DIR):
        directory.mkdir(parents=True, exist_ok=True)


def new_run_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def create_ocr_run_dir() -> tuple[str, Path]:
    ensure_runtime_directories()
    run_id = new_run_id("ocr")
    run_dir = OCR_RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_id, run_dir


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


def guess_artifact_kind(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
        return "image"
    if suffix in {".json"}:
        return "json"
    if suffix in {".txt", ".log", ".md"}:
        return "text"
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type and mime_type.startswith("image/"):
        return "image"
    return "file"


def artifact_descriptor(root_dir: Path, public_prefix: str, path: Path, label: str | None = None) -> dict[str, str]:
    relative_path = path.relative_to(root_dir).as_posix()
    return {
        "label": label or path.name,
        "name": path.name,
        "kind": guess_artifact_kind(path),
        "relative_path": relative_path,
        "url": f"{public_prefix}/{relative_path}",
    }


def bundle_payload(
    *,
    summary: str,
    artifacts: Iterable[dict[str, str]] | None = None,
    text: str | None = None,
    data: Any | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"summary": summary}
    if artifacts:
        payload["artifacts"] = list(artifacts)
    if text is not None:
        payload["text"] = text
    if data is not None:
        payload["data"] = data
    return payload


def ensure_safe_relative_path(base_dir: Path, relative_path: str) -> Path:
    candidate = (base_dir / relative_path).resolve()
    base_resolved = base_dir.resolve()
    if base_resolved == candidate or base_resolved in candidate.parents:
        return candidate
    raise ValueError("Path escapes the runtime directory.")
