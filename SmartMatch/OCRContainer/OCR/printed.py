from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List

os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

import paddle
from paddleocr import PaddleOCR

CONFIG_PATH = Path(__file__).with_name("config.json")
with open(CONFIG_PATH, "r", encoding="utf-8") as handle:
    CONFIG = json.load(handle)


def _printed_model_config() -> dict[str, str]:
    configured = dict(CONFIG.get("printed_models", {}))
    env_overrides = {
        "text_detection_model_name": os.getenv("SMARTMATCH_PRINTED_TEXT_DETECTION_MODEL_NAME"),
        "text_recognition_model_name": os.getenv("SMARTMATCH_PRINTED_TEXT_RECOGNITION_MODEL_NAME"),
        "doc_orientation_classify_model_name": os.getenv("SMARTMATCH_PRINTED_DOC_ORIENTATION_MODEL_NAME"),
        "doc_unwarping_model_name": os.getenv("SMARTMATCH_PRINTED_DOC_UNWARPING_MODEL_NAME"),
        "textline_orientation_model_name": os.getenv("SMARTMATCH_PRINTED_TEXTLINE_ORIENTATION_MODEL_NAME"),
    }
    for key, value in env_overrides.items():
        if value:
            configured[key] = value
    return configured


def _to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "json"):
        return _to_jsonable(obj.json)
    if isinstance(obj, dict):
        return {str(key): _to_jsonable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(item) for item in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            pass
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    return obj


class PrintedOCR:
    def __init__(self) -> None:
        self.device = (
            "gpu:0"
            if (paddle.device.is_compiled_with_cuda() and paddle.device.cuda.device_count() > 0)
            else "cpu"
        )
        self.engine = PaddleOCR(
            device=self.device,
            use_doc_orientation_classify=True,
            use_doc_unwarping=True,
            use_textline_orientation=True,
            **_printed_model_config(),
        )

    def recognize(self, image_path: str) -> Dict[str, Any]:
        results = list(self.engine.predict(str(image_path)))
        if not results:
            return {
                "status": "done",
                "engine": "paddleocr",
                "device": self.device,
                "text": "",
                "confidence": None,
                "lines": [],
            }

        payload = _to_jsonable(results[0])
        payload = payload.get("res", payload)
        texts: List[str] = payload.get("rec_texts") or []
        scores: List[float] = payload.get("rec_scores") or []
        boxes = payload.get("rec_boxes") or payload.get("dt_polys") or []

        lines = []
        for index, text in enumerate(texts):
            lines.append(
                {
                    "index": index + 1,
                    "text": text,
                    "confidence": float(scores[index]) if index < len(scores) else None,
                    "box": boxes[index] if index < len(boxes) else None,
                }
            )

        confidence = float(sum(scores) / len(scores)) if scores else None
        return {
            "status": "done",
            "engine": "paddleocr",
            "device": self.device,
            "text": "\n".join(text.strip() for text in texts if text and text.strip()),
            "confidence": confidence,
            "lines": lines,
        }
