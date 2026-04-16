from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict

from handwritten import HandwrittenOCR
from printed import PrintedOCR

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    CONFIG = json.load(handle)

_PRINTED_OCR: PrintedOCR | None = None
_HANDWRITTEN_OCR: HandwrittenOCR | None = None


def _get_printed_ocr() -> PrintedOCR:
    global _PRINTED_OCR
    if _PRINTED_OCR is None:
        _PRINTED_OCR = PrintedOCR()
    return _PRINTED_OCR


def _get_handwritten_ocr() -> HandwrittenOCR:
    global _HANDWRITTEN_OCR
    if _HANDWRITTEN_OCR is None:
        _HANDWRITTEN_OCR = HandwrittenOCR()
    return _HANDWRITTEN_OCR


def get_available_handwritten_models() -> list[str]:
    try:
        return _get_handwritten_ocr().available_models()
    except FileNotFoundError:
        return []


def get_default_handwritten_model() -> str:
    configured_default = str(
        os.getenv("SMARTMATCH_DEFAULT_HANDWRITTEN_MODEL", CONFIG.get("default_handwritten_model", "best"))
    ).strip().lower()
    available_models = get_available_handwritten_models()
    if configured_default in available_models:
        return configured_default
    if available_models:
        return available_models[0]
    return configured_default


def recognize_segment(image_path: str, label: int, handwritten_model: str | None = None) -> Dict:
    handwritten_label = int(os.getenv("SMARTMATCH_OCR_HANDWRITTEN_LABEL", CONFIG.get("handwritten_label", 1)))
    selected_handwritten_model = handwritten_model or get_default_handwritten_model()
    if int(label) == handwritten_label:
        return _get_handwritten_ocr().recognize(image_path, model_name=selected_handwritten_model)
    return _get_printed_ocr().recognize(image_path)
