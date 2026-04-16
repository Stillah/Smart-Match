from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from runtime import TRAINING_JOBS_DIR, ensure_runtime_directories, ensure_safe_relative_path
from training_service import get_training_job, list_trainable_models, list_training_jobs, start_training_job

ensure_runtime_directories()

app = FastAPI(
    title="Smart Match Training API",
    summary="Training microservice for Smart Match OCR models.",
    description="Launch and inspect OCR-related training jobs as a standalone microservice.",
    version="1.0.0",
)


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")


class TrainingJobRequest(BaseModel):
    model_key: str = Field(description="Training model key returned by /api/training/models.")
    parameters: dict[str, Any] = Field(default_factory=dict, description="Training parameters for the selected model.")


class TrainingJobLogResponse(BaseModel):
    job_id: str
    status: str
    log: str


COMMON_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "Invalid request payload."},
    404: {"model": ErrorResponse, "description": "Resource not found."},
    500: {"model": ErrorResponse, "description": "The service failed while processing the request."},
}


@app.get("/", tags=["Service"])
def root() -> dict[str, Any]:
    return {
        "service": "smartmatch-training",
        "status": "ok",
        "jobs_dir": str(TRAINING_JOBS_DIR),
        "models": list_trainable_models(),
    }


@app.get("/health", tags=["Service"])
@app.get("/api/training/health", tags=["Service"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "smartmatch-training"}


@app.get("/api/training/models", tags=["Training"])
def get_training_models() -> list[dict[str, Any]]:
    return list_trainable_models()


@app.get("/api/training/jobs", tags=["Training"])
def get_all_training_jobs() -> list[dict[str, Any]]:
    return list_training_jobs()


@app.post(
    "/api/training/jobs",
    tags=["Training"],
    responses=COMMON_ERROR_RESPONSES,
)
def create_training_job(request: TrainingJobRequest) -> dict[str, Any]:
    try:
        return start_training_job(request.model_key, request.parameters)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get(
    "/api/training/jobs/{job_id}",
    tags=["Training"],
    responses=COMMON_ERROR_RESPONSES,
)
def get_training_job_status(job_id: str) -> dict[str, Any]:
    try:
        return get_training_job(job_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get(
    "/api/training/jobs/{job_id}/log",
    tags=["Training"],
    response_model=TrainingJobLogResponse,
    responses=COMMON_ERROR_RESPONSES,
)
def get_training_job_log(job_id: str) -> JSONResponse:
    job = get_training_job_status(job_id)
    return JSONResponse({"job_id": job_id, "status": job["status"], "log": job.get("log_tail", "")})


@app.get(
    "/api/training/jobs/{job_id}/files/{relative_path:path}",
    tags=["Training"],
    responses=COMMON_ERROR_RESPONSES,
)
def get_training_job_file(job_id: str, relative_path: str) -> FileResponse:
    job_dir = TRAINING_JOBS_DIR / job_id
    if not job_dir.exists():
        raise HTTPException(status_code=404, detail=f"Training job not found: {job_id}")
    try:
        target_path = ensure_safe_relative_path(job_dir, relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")
    return FileResponse(target_path)
