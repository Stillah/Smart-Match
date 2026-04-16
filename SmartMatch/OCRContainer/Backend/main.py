from __future__ import annotations

import json
import os
import shutil
import tempfile
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, Path as FastAPIPath, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field

from document_schema import build_assignment_document, save_assignment_document
from ocr_service import (
    STAGE_ORDER,
    get_available_handwritten_models,
    get_default_handwritten_model,
    list_ocr_stages,
    run_selected_stages,
)
from runtime import OCR_RUNS_DIR, RUNTIME_ROOT, ensure_runtime_directories, ensure_safe_relative_path

ensure_runtime_directories()

ALLOWED_IMAGE_EXTENSIONS = {
    item.strip().lower()
    for item in os.getenv("SMARTMATCH_ALLOWED_IMAGE_EXTENSIONS", ".jpg,.jpeg,.png,.tif,.tiff,.bmp,.webp").split(",")
    if item.strip()
}

API_TAGS = [
    {
        "name": "Service",
        "description": "Health checks and service metadata.",
    },
    {
        "name": "OCR",
        "description": "Structured document extraction and stage-driven OCR execution.",
    },
]

app = FastAPI(
    title="Smart Match OCR API",
    summary="OCR pipeline microservice for Smart Match.",
    description=(
        "Runs the Smart Match OCR pipeline as a standalone microservice. "
        "The service can execute individual OCR stages for debugging and expose "
        "a production-oriented document extraction endpoint that returns structured JSON."
    ),
    version="1.0.0",
    openapi_tags=API_TAGS,
)


class ErrorResponse(BaseModel):
    detail: str = Field(description="Human-readable error message.")


class OCRStageDescriptor(BaseModel):
    name: str = Field(description="Internal stage key.")
    title: str = Field(description="Display name shown in the frontend.")
    input_kind: str = Field(description="Manual input type expected by the stage.")
    description: str = Field(description="Short English description of the stage behavior.")


class OCRConfigResponse(BaseModel):
    ocr_stages: list[OCRStageDescriptor] = Field(description="Stage metadata used by the OCR console.")
    handwritten_models: list[str] = Field(description="Available handwritten OCR model choices.")
    allowed_extensions: list[str] = Field(description="Allowed source image extensions.")


COMMON_ERROR_RESPONSES = {
    400: {"model": ErrorResponse, "description": "The request payload is invalid for the selected operation."},
    404: {"model": ErrorResponse, "description": "The requested runtime resource was not found."},
    500: {"model": ErrorResponse, "description": "The server failed while running the selected OCR operation."},
}


def _parse_stage_selection(raw_value: str | None) -> list[str]:
    if not raw_value:
        return STAGE_ORDER.copy()
    raw_value = raw_value.strip()
    if raw_value.startswith("["):
        try:
            parsed = json.loads(raw_value)
        except json.JSONDecodeError as exc:
            raise ValueError("Stages payload must be valid JSON or a comma-separated list.") from exc
        if not isinstance(parsed, list):
            raise ValueError("Stages payload must be a list.")
        return [str(item) for item in parsed]
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def _validate_image_path(path: Path) -> None:
    if path.suffix.lower() not in ALLOWED_IMAGE_EXTENSIONS:
        raise ValueError(f"Unsupported image extension: {path.suffix}")
    try:
        with Image.open(path) as image:
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Invalid or corrupt image: {path.name}") from exc


async def _save_uploads_temporarily(files: list[UploadFile]) -> tuple[Path, list[Path]]:
    temp_dir = Path(tempfile.mkdtemp(prefix="smartmatch_upload_", dir=RUNTIME_ROOT))
    saved_paths = []
    for index, file in enumerate(files, start=1):
        filename = file.filename or f"input_{index}"
        target_path = temp_dir / f"{index:03d}_{Path(filename).name}"
        with open(target_path, "wb") as handle:
            handle.write(await file.read())
        _validate_image_path(target_path)
        saved_paths.append(target_path)
    return temp_dir, saved_paths


def _ocr_config() -> dict[str, Any]:
    default_handwritten_model = get_default_handwritten_model()
    handwritten_models = []
    for model_name in [default_handwritten_model, "best", *get_available_handwritten_models()]:
        if model_name not in handwritten_models:
            handwritten_models.append(model_name)
    return {
        "ocr_stages": list_ocr_stages(),
        "handwritten_models": handwritten_models,
        "allowed_extensions": sorted(ALLOWED_IMAGE_EXTENSIONS),
    }


def _handle_ocr_exception(exc: Exception) -> HTTPException:
    if isinstance(exc, FileNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=str(exc))
    if isinstance(exc, RuntimeError):
        return HTTPException(status_code=500, detail=str(exc))
    return HTTPException(status_code=500, detail="Unexpected OCR service failure.")


@app.get("/", tags=["Service"])
def root() -> dict[str, Any]:
    return {
        "service": "smartmatch-ocr",
        "status": "ok",
        "runtime_dir": str(RUNTIME_ROOT),
        "stages": STAGE_ORDER,
        "process_endpoint": "/api/ocr/process",
        "debug_endpoint": "/api/ocr/run",
    }


@app.get("/health", tags=["Service"])
@app.get("/api/ocr/health", tags=["Service"])
def health() -> dict[str, str]:
    return {"status": "ok", "service": "smartmatch-ocr"}


@app.get(
    "/api/ocr/config",
    tags=["OCR"],
    summary="Get OCR frontend configuration",
    response_model=OCRConfigResponse,
)
def get_ocr_config() -> OCRConfigResponse:
    return _ocr_config()


@app.post(
    "/api/ocr/process",
    tags=["OCR"],
    summary="Run the full OCR pipeline and return structured JSON",
    responses=COMMON_ERROR_RESPONSES,
)
async def process_document(
    handwritten_model: Annotated[
        str,
        Form(description="Handwritten OCR model selection. Defaults to the configured default model."),
    ] = get_default_handwritten_model(),
    include_debug: Annotated[
        bool,
        Form(description="Include the full stage-by-stage debug payload in the JSON response."),
    ] = False,
    files: Annotated[
        list[UploadFile],
        File(description="Exactly one metrical-book page image (.jpg, .png, .tiff, etc.)."),
    ] = ...,
):
    temp_dir: Path | None = None
    saved_paths: list[Path] = []
    try:
        temp_dir, saved_paths = await _save_uploads_temporarily(files)
        if len(saved_paths) != 1:
            raise ValueError("Structured document extraction expects exactly one source image.")

        summary = run_selected_stages(
            selected_stages=STAGE_ORDER.copy(),
            input_files=saved_paths,
            text_input=None,
            handwritten_model=handwritten_model,
            manual_segment_label=None,
        )
        run_dir = OCR_RUNS_DIR / summary["run_id"]
        document = build_assignment_document(summary, run_dir)
        save_assignment_document(run_dir, document)
        if include_debug:
            document["pipeline_debug"] = summary
        return document
    except Exception as exc:  # pragma: no cover - fastapi exception mapping
        raise _handle_ocr_exception(exc) from exc
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post(
    "/api/ocr/run",
    tags=["OCR"],
    summary="Run a selected OCR stage slice",
    responses=COMMON_ERROR_RESPONSES,
)
async def run_ocr_pipeline(
    stages: Annotated[
        str | None,
        Form(
            description=(
                "JSON array or comma-separated list of stage names. "
                "The selected stages must form one continuous OCR pipeline slice. "
                "Defaults to the full pipeline."
            )
        ),
    ] = None,
    text_input: Annotated[
        str | None,
        Form(description="Manual text input. Required when the first selected stage is NER."),
    ] = None,
    handwritten_model: Annotated[
        str,
        Form(
            description=(
                "Handwritten OCR model selection. Defaults to the configured default model. "
                "Use 'best' to evaluate all available handwritten models and keep the best candidate."
            )
        ),
    ] = get_default_handwritten_model(),
    manual_segment_label: Annotated[
        str | None,
        Form(description="Required when OCR is the first selected stage. Use 'printed' or 'handwritten'."),
    ] = None,
    files: Annotated[
        list[UploadFile] | None,
        File(
            description=(
                "Input files for the first selected stage. Upload one image for preprocess or segment, "
                "or one or more segment images for classify or OCR."
            )
        ),
    ] = None,
):
    selected_stages = _parse_stage_selection(stages)
    uploaded_files = files or []
    temp_dir: Path | None = None
    saved_paths: list[Path] = []

    try:
        if uploaded_files:
            temp_dir, saved_paths = await _save_uploads_temporarily(uploaded_files)
        return run_selected_stages(
            selected_stages=selected_stages,
            input_files=saved_paths,
            text_input=text_input,
            handwritten_model=handwritten_model,
            manual_segment_label=manual_segment_label,
        )
    except Exception as exc:  # pragma: no cover - fastapi exception mapping
        raise _handle_ocr_exception(exc) from exc
    finally:
        if temp_dir and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.post(
    "/api/ocr/stages/{stage_name}",
    tags=["OCR"],
    summary="Run one OCR stage",
    responses=COMMON_ERROR_RESPONSES,
)
async def run_single_ocr_stage(
    stage_name: Annotated[
        str,
        FastAPIPath(description="One OCR stage name: preprocess, segment, classify, ocr, or ner."),
    ],
    text_input: Annotated[
        str | None,
        Form(description="Manual text input. Required when the selected stage is NER."),
    ] = None,
    handwritten_model: Annotated[
        str,
        Form(description="Handwritten OCR model selection. Defaults to the configured default model."),
    ] = get_default_handwritten_model(),
    manual_segment_label: Annotated[
        str | None,
        Form(description="Manual segment label for direct OCR stage execution."),
    ] = None,
    files: Annotated[
        list[UploadFile] | None,
        File(description="Input files for the selected stage."),
    ] = None,
):
    return await run_ocr_pipeline(
        stages=stage_name,
        text_input=text_input,
        handwritten_model=handwritten_model,
        manual_segment_label=manual_segment_label,
        files=files,
    )


@app.get(
    "/api/ocr/runs/{run_id}/summary",
    tags=["OCR"],
    summary="Get OCR run summary",
    responses=COMMON_ERROR_RESPONSES,
)
def get_ocr_run_summary(
    run_id: Annotated[str, FastAPIPath(description="OCR run identifier returned by the execution endpoint.")],
) -> JSONResponse:
    summary_path = OCR_RUNS_DIR / run_id / "summary.json"
    if not summary_path.exists():
        raise HTTPException(status_code=404, detail=f"OCR run not found: {run_id}")
    return JSONResponse(json.loads(summary_path.read_text(encoding="utf-8")))


@app.get(
    "/api/ocr/runs/{run_id}/files/{relative_path:path}",
    tags=["OCR"],
    summary="Get OCR run artifact",
    responses=COMMON_ERROR_RESPONSES,
)
def get_ocr_run_file(
    run_id: Annotated[str, FastAPIPath(description="OCR run identifier.")],
    relative_path: Annotated[str, FastAPIPath(description="Artifact path relative to the OCR run directory.")],
) -> FileResponse:
    run_dir = OCR_RUNS_DIR / run_id
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"OCR run not found: {run_id}")
    try:
        target_path = ensure_safe_relative_path(run_dir, relative_path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"File not found: {relative_path}")
    return FileResponse(target_path)


@app.post(
    "/process",
    tags=["OCR"],
    summary="Run the full OCR pipeline",
    responses=COMMON_ERROR_RESPONSES,
)
async def legacy_process(
    file: Annotated[UploadFile, File(description="Single source image for the full OCR pipeline.")],
    handwritten_model: Annotated[
        str,
        Form(description="Handwritten OCR model selection. Defaults to the configured default model."),
    ] = get_default_handwritten_model(),
):
    return await process_document(
        handwritten_model=handwritten_model,
        include_debug=False,
        files=[file],
    )
