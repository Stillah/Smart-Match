from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime import CONTAINER_ROOT, create_ocr_run_dir, artifact_descriptor, bundle_payload, save_json

for module_dir in (
    CONTAINER_ROOT / "ContentClassification",
    CONTAINER_ROOT / "OCR",
    CONTAINER_ROOT / "NER",
):
    module_dir_str = str(module_dir)
    if module_dir_str not in sys.path:
        sys.path.insert(0, module_dir_str)

from classify import classify_images  # noqa: E402
from ner_pipeline import extract_entities  # noqa: E402
from ocr_pipeline import get_available_handwritten_models, get_default_handwritten_model, recognize_segment  # noqa: E402

STAGE_ORDER = ["preprocess", "segment", "classify", "ocr", "ner"]
STAGE_METADATA = {item["name"]: item for item in (
    {
        "name": "preprocess",
        "title": "Preprocess",
        "input_kind": "image",
        "description": "Runs Kraken binarization on the uploaded image.",
    },
    {
        "name": "segment",
        "title": "Segmentation",
        "input_kind": "image",
        "description": "Splits a preprocessed table image into ordered column segments.",
    },
    {
        "name": "classify",
        "title": "Segment Classification",
        "input_kind": "segments",
        "description": "Predicts whether each segment is printed or handwritten.",
    },
    {
        "name": "ocr",
        "title": "OCR",
        "input_kind": "segments",
        "description": "Runs PaddleOCR for printed segments or TrOCR for handwritten ones.",
    },
    {
        "name": "ner",
        "title": "NER",
        "input_kind": "text",
        "description": "Runs Natasha on the assembled OCR text.",
    },
)}
IMAGE_INPUT_STAGES = {"preprocess", "segment"}
SEGMENT_INPUT_STAGES = {"classify", "ocr"}


def list_ocr_stages() -> list[dict[str, Any]]:
    return [STAGE_METADATA[stage_name] for stage_name in STAGE_ORDER]


def _stage_index(stage_name: str) -> int:
    return STAGE_ORDER.index(stage_name)


def _validate_stage_sequence(stage_names: list[str]) -> list[str]:
    if not stage_names:
        raise ValueError("At least one stage must be selected.")
    cleaned = []
    seen = set()
    for stage_name in stage_names:
        if stage_name not in STAGE_ORDER:
            raise ValueError(f"Unknown stage: {stage_name}")
        if stage_name in seen:
            continue
        cleaned.append(stage_name)
        seen.add(stage_name)

    order_indexes = [_stage_index(stage_name) for stage_name in cleaned]
    if order_indexes != sorted(order_indexes):
        raise ValueError("Selected stages must follow the pipeline order.")
    if order_indexes != list(range(order_indexes[0], order_indexes[0] + len(order_indexes))):
        raise ValueError("Selected stages must form a continuous pipeline slice.")
    return cleaned


def _run_command(name: str, cmd: list[str]) -> None:
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(CONTAINER_ROOT),
        env={**os.environ},
    )
    if result.returncode != 0:
        error_message = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"{name} failed: {error_message}")


def _copy_manual_inputs(input_files: list[Path], target_dir: Path) -> list[Path]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied_paths = []
    for index, input_path in enumerate(input_files, start=1):
        filename = f"{index:03d}_{input_path.name}"
        target_path = target_dir / filename
        shutil.copy2(input_path, target_path)
        copied_paths.append(target_path)
    return copied_paths


def _artifacts_for_paths(run_id: str, run_dir: Path, paths: list[Path], label_prefix: str) -> list[dict[str, str]]:
    public_prefix = f"/api/ocr/runs/{run_id}/files"
    return [
        artifact_descriptor(run_dir, public_prefix, path, label=f"{label_prefix} {index}")
        for index, path in enumerate(paths, start=1)
    ]


def _image_payload(run_id: str, run_dir: Path, image_path: Path, summary: str, label: str) -> dict[str, Any]:
    return bundle_payload(
        summary=summary,
        artifacts=_artifacts_for_paths(run_id, run_dir, [image_path], label),
    )


def _segment_payload(
    run_id: str,
    run_dir: Path,
    segment_paths: list[Path],
    summary: str,
    data: Any | None = None,
) -> dict[str, Any]:
    return bundle_payload(
        summary=summary,
        artifacts=_artifacts_for_paths(run_id, run_dir, segment_paths, "Segment"),
        data=data,
    )


def _classification_rows_from_context(context: dict[str, Any]) -> list[dict[str, Any]]:
    rows = context.get("classification_rows")
    if rows:
        return rows

    segment_paths = [Path(path) for path in context.get("segment_paths", [])]
    classifications = context.get("classifications", {})
    derived_rows = []
    for index, segment_path in enumerate(segment_paths, start=1):
        label_value = int(classifications.get(str(segment_path), 0))
        derived_rows.append(
            {
                "index": index,
                "filename": segment_path.name,
                "classification_label": label_value,
                "classification_name": "handwritten" if label_value == 1 else "printed",
            }
        )
    return derived_rows


def _stage_input_payload(run_id: str, run_dir: Path, stage_name: str, context: dict[str, Any]) -> dict[str, Any]:
    if stage_name in IMAGE_INPUT_STAGES:
        image_path = Path(context["image_path"])
        summary = "Input image" if stage_name == "preprocess" else "Preprocessed image"
        label = "Input image" if stage_name == "preprocess" else "Preprocessed image"
        return _image_payload(run_id, run_dir, image_path, summary, label)

    if stage_name == "classify":
        segment_paths = [Path(path) for path in context["segment_paths"]]
        manifest = context.get("segment_manifest")
        return _segment_payload(
            run_id,
            run_dir,
            segment_paths,
            f"Segment input ({len(segment_paths)} file(s))",
            manifest,
        )

    if stage_name == "ocr":
        segment_paths = [Path(path) for path in context["segment_paths"]]
        rows = _classification_rows_from_context(context)
        return _segment_payload(
            run_id,
            run_dir,
            segment_paths,
            f"OCR input ({len(segment_paths)} segment(s))",
            rows,
        )

    if stage_name == "ner":
        raw_text = context.get("raw_text", "")
        return bundle_payload(summary="OCR text input", text=raw_text)

    raise ValueError(f"Unknown stage: {stage_name}")


def _manual_input_payload(
    *,
    run_id: str,
    run_dir: Path,
    first_stage: str,
    input_paths: list[Path],
    text_input: str | None,
    manual_segment_label: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    context: dict[str, Any] = {}
    if first_stage in IMAGE_INPUT_STAGES:
        if len(input_paths) != 1:
            raise ValueError(f"Stage '{first_stage}' expects exactly one image file.")
        context["image_path"] = input_paths[0]
        return context, bundle_payload(
            summary="Manual image input",
            artifacts=_artifacts_for_paths(run_id, run_dir, input_paths, "Input image"),
        )

    if first_stage == "classify":
        if not input_paths:
            raise ValueError("Classification requires one or more segment images.")
        context["segment_paths"] = input_paths
        return context, bundle_payload(
            summary=f"Manual segment input ({len(input_paths)} file(s))",
            artifacts=_artifacts_for_paths(run_id, run_dir, input_paths, "Segment"),
        )

    if first_stage == "ocr":
        if not input_paths:
            raise ValueError("OCR requires one or more segment images.")
        if manual_segment_label not in {"printed", "handwritten"}:
            raise ValueError("OCR without classification requires manual_segment_label.")
        context["segment_paths"] = input_paths
        label_value = 1 if manual_segment_label == "handwritten" else 0
        context["classifications"] = {str(path): label_value for path in input_paths}
        context["classification_rows"] = [
            {
                "filename": path.name,
                "classification_label": label_value,
                "classification_name": manual_segment_label,
            }
            for path in input_paths
        ]
        return context, bundle_payload(
            summary=f"Manual OCR input ({len(input_paths)} segment(s), {manual_segment_label})",
            artifacts=_artifacts_for_paths(run_id, run_dir, input_paths, "Segment"),
            data=context["classification_rows"],
        )

    if first_stage == "ner":
        if not text_input:
            raise ValueError("NER requires a text input.")
        context["raw_text"] = text_input
        return context, bundle_payload(
            summary="Manual text input",
            text=text_input,
        )

    raise ValueError(f"Unsupported first stage: {first_stage}")


def _run_preprocess(run_id: str, run_dir: Path, stage_dir: Path, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_path = Path(context["image_path"])
    output_dir = stage_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = CONTAINER_ROOT / "binarization.sh"
    _run_command("Preprocess", ["bash", str(script_path), str(input_path), str(output_dir)])
    output_path = output_dir / input_path.name

    output_payload = bundle_payload(
        summary="Binarized image",
        artifacts=_artifacts_for_paths(run_id, run_dir, [output_path], "Binarized image"),
    )
    context["image_path"] = output_path
    return output_payload, context


def _run_segmentation(run_id: str, run_dir: Path, stage_dir: Path, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    input_path = Path(context["image_path"])
    output_dir = stage_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    script_path = CONTAINER_ROOT / "LayoutDetection" / "detection.sh"
    _run_command("Segmentation", ["bash", str(script_path), str(input_path), str(output_dir)])

    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    segment_paths = [output_dir / item["filename"] for item in manifest]
    context["segment_paths"] = segment_paths
    context["segment_manifest"] = manifest

    output_payload = bundle_payload(
        summary=f"Generated {len(segment_paths)} segment(s)",
        artifacts=_artifacts_for_paths(run_id, run_dir, segment_paths, "Segment"),
        data=manifest,
    )
    return output_payload, context


def _run_classification(run_id: str, run_dir: Path, stage_dir: Path, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    segment_paths = [Path(path) for path in context["segment_paths"]]
    results = classify_images([str(path) for path in segment_paths])

    rows = []
    for index, segment_path in enumerate(segment_paths, start=1):
        label_value = int(results[str(segment_path)])
        rows.append(
            {
                "index": index,
                "filename": segment_path.name,
                "classification_label": label_value,
                "classification_name": "handwritten" if label_value == 1 else "printed",
            }
        )

    output_json_path = stage_dir / "output" / "classification.json"
    save_json(output_json_path, rows)
    context["classifications"] = results
    context["classification_rows"] = rows

    output_payload = bundle_payload(
        summary=f"Classified {len(rows)} segment(s)",
        artifacts=[artifact_descriptor(run_dir, f"/api/ocr/runs/{run_id}/files", output_json_path, "Classification JSON")],
        data=rows,
    )
    return output_payload, context


def _run_ocr(
    run_id: str,
    run_dir: Path,
    stage_dir: Path,
    context: dict[str, Any],
    handwritten_model: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    segment_paths = [Path(path) for path in context["segment_paths"]]
    classifications: dict[str, int] = context.get("classifications", {})
    rows = context.get("classification_rows", [])

    results = []
    raw_text_parts = []
    for index, segment_path in enumerate(segment_paths, start=1):
        label_value = int(classifications.get(str(segment_path), 0))
        ocr_result = recognize_segment(
            str(segment_path),
            label_value,
            handwritten_model=handwritten_model,
        )
        text = (ocr_result.get("text") or "").strip()
        if text:
            raw_text_parts.append(text)
        results.append(
            {
                "index": index,
                "filename": segment_path.name,
                "classification_label": label_value,
                "classification_name": "handwritten" if label_value == 1 else "printed",
                "ocr": ocr_result,
            }
        )

    raw_text = "\n".join(raw_text_parts).strip()
    output_dir = stage_dir / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_text_path = output_dir / "raw_text.txt"
    raw_text_path.write_text(raw_text, encoding="utf-8")
    output_json_path = output_dir / "ocr.json"
    save_json(output_json_path, results)

    context["ocr_results"] = results
    context["raw_text"] = raw_text

    output_payload = bundle_payload(
        summary=f"OCR completed for {len(results)} segment(s)",
        artifacts=[
            artifact_descriptor(run_dir, f"/api/ocr/runs/{run_id}/files", raw_text_path, "Raw text"),
            artifact_descriptor(run_dir, f"/api/ocr/runs/{run_id}/files", output_json_path, "OCR JSON"),
        ],
        text=raw_text,
        data={"segments": results, "classification_rows": rows},
    )
    return output_payload, context


def _run_ner(run_id: str, run_dir: Path, stage_dir: Path, context: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_text = context.get("raw_text", "")
    entities = extract_entities(raw_text)

    output_json_path = stage_dir / "output" / "ner.json"
    save_json(output_json_path, entities)
    context["ner_result"] = entities

    output_payload = bundle_payload(
        summary="Named entities extracted",
        artifacts=[artifact_descriptor(run_dir, f"/api/ocr/runs/{run_id}/files", output_json_path, "NER JSON")],
        text=raw_text,
        data=entities,
    )
    return output_payload, context


def run_selected_stages(
    *,
    selected_stages: list[str],
    input_files: list[Path],
    text_input: str | None,
    handwritten_model: str | None = None,
    manual_segment_label: str | None = None,
) -> dict[str, Any]:
    handwritten_model = handwritten_model or get_default_handwritten_model()
    stage_names = _validate_stage_sequence(selected_stages)
    if handwritten_model not in {"best", *get_available_handwritten_models()}:
        raise ValueError(f"Unsupported handwritten model selection: {handwritten_model}")

    started_at = datetime.now(timezone.utc)
    started_monotonic = time.perf_counter()
    run_id, run_dir = create_ocr_run_dir()
    manual_input_dir = run_dir / "manual_input"
    copied_input_paths = _copy_manual_inputs(input_files, manual_input_dir) if input_files else []

    first_stage = stage_names[0]
    context, first_input = _manual_input_payload(
        run_id=run_id,
        run_dir=run_dir,
        first_stage=first_stage,
        input_paths=copied_input_paths,
        text_input=text_input,
        manual_segment_label=manual_segment_label,
    )

    stage_results = []
    for position, stage_name in enumerate(stage_names, start=1):
        stage_dir = run_dir / f"{position:02d}_{stage_name}"
        input_payload = first_input if position == 1 else _stage_input_payload(run_id, run_dir, stage_name, context)
        if stage_name == "preprocess":
            output_payload, context = _run_preprocess(run_id, run_dir, stage_dir, context)
        elif stage_name == "segment":
            output_payload, context = _run_segmentation(run_id, run_dir, stage_dir, context)
        elif stage_name == "classify":
            output_payload, context = _run_classification(run_id, run_dir, stage_dir, context)
        elif stage_name == "ocr":
            output_payload, context = _run_ocr(run_id, run_dir, stage_dir, context, handwritten_model)
        elif stage_name == "ner":
            output_payload, context = _run_ner(run_id, run_dir, stage_dir, context)
        else:  # pragma: no cover - impossible after validation
            raise ValueError(f"Unknown stage: {stage_name}")

        stage_result = {
            "stage": stage_name,
            "title": STAGE_METADATA[stage_name]["title"],
            "input": input_payload,
            "output": output_payload,
        }
        stage_results.append(stage_result)

    completed_at = datetime.now(timezone.utc)
    duration_seconds = round(time.perf_counter() - started_monotonic, 3)
    summary = {
        "run_id": run_id,
        "selected_stages": stage_names,
        "handwritten_model": handwritten_model,
        "manual_segment_label": manual_segment_label,
        "started_at": started_at.isoformat().replace("+00:00", "Z"),
        "completed_at": completed_at.isoformat().replace("+00:00", "Z"),
        "duration_seconds": duration_seconds,
        "stage_results": stage_results,
    }
    save_json(run_dir / "summary.json", summary)
    return summary
