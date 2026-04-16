import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

from dataloader import SimpleImageDataset, TARGET_SIZE as DEFAULT_TARGET_SIZE
from model import SimpleCNN

_config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(_config_path, "r") as handle:
    _config = json.load(handle)
DEFAULT_WEIGHTS_PATH = _config.get("weights_path", "../../../models/classifier.pth")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _candidate_weight_paths(raw_path: Optional[str]) -> list[str]:
    if not raw_path:
        return []

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
        resolved_candidates.append(candidate_str)
        seen.add(candidate_str)
    return resolved_candidates


def _resolve_weights_path(weights_path: Optional[str] = None) -> str:
    raw_candidates = [
        weights_path,
        os.getenv("SMARTMATCH_CLASSIFIER_WEIGHTS"),
        DEFAULT_WEIGHTS_PATH,
        str(PROJECT_ROOT / "models" / "classifier.pth"),
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "models", "classifier.pth"),
    ]
    candidates = []
    for raw_candidate in raw_candidates:
        candidates.extend(_candidate_weight_paths(raw_candidate))

    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Classifier weights not found. Set SMARTMATCH_CLASSIFIER_WEIGHTS or provide weights_path."
    )


def _load_model(
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> SimpleCNN:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SimpleCNN()
    path = _resolve_weights_path(weights_path)
    model.load_state_dict(torch.load(path, map_location=device))
    model.to(device)
    model.eval()
    return model


def classify_images(
    image_paths: List[str],
    model: Optional[SimpleCNN] = None,
    weights_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    target_size: tuple = DEFAULT_TARGET_SIZE,
    threshold: float | None = None,
    batch_size: int | None = None,
) -> Dict[str, int]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if threshold is None:
        threshold = float(os.getenv("SMARTMATCH_CLASSIFIER_THRESHOLD", "0.5"))
    if batch_size is None:
        batch_size = int(os.getenv("SMARTMATCH_CLASSIFIER_BATCH_SIZE", "32"))

    if model is None:
        model = _load_model(weights_path, device)
    else:
        model.eval()
        model.to(device)

    dataset = SimpleImageDataset(image_paths, target_size=target_size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    results: Dict[str, int] = {}
    with torch.no_grad():
        for images, paths in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.sigmoid(logits)
            preds = (probs >= threshold).long().cpu().tolist()
            for path, pred in zip(paths, preds):
                results[path] = pred

    return results
