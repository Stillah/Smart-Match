import json
import logging
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

_config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(_config_path, "r") as handle:
    _config = json.load(handle)

TARGET_SIZE = tuple(_config["target_size"])
TARGET_SIZE = (
    int(os.getenv("SMARTMATCH_CLASSIFIER_TARGET_HEIGHT", TARGET_SIZE[0])),
    int(os.getenv("SMARTMATCH_CLASSIFIER_TARGET_WIDTH", TARGET_SIZE[1])),
)
_runtime_root = os.getenv(
    "SMARTMATCH_RUNTIME_DIR",
    os.path.join(os.path.dirname(os.path.dirname(__file__)), ".runtime"),
)
_log_dir = os.path.join(_runtime_root, "logs")
os.makedirs(_log_dir, exist_ok=True)

logger = logging.getLogger("classification")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.FileHandler(os.path.join(_log_dir, "classification.log"))
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(message)s"))
    logger.addHandler(handler)


class SimpleImageDataset(Dataset):
    def __init__(self, file_paths, target_size=TARGET_SIZE):
        self.target_size = target_size
        self.file_paths = []
        for path in file_paths:
            try:
                with Image.open(path) as img:
                    img.verify()
                self.file_paths.append(path)
            except Exception as exc:
                logger.info(f"[SKIP] Invalid image: {path} - {exc}")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        try:
            img = Image.open(img_path).convert("L")
            img = img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)
            img_np = np.array(img, dtype=np.float32) / 255.0
            tensor = torch.from_numpy(img_np).unsqueeze(0)
            return tensor, img_path
        except Exception as exc:
            logger.info(f"[SKIP] Failed to load: {img_path} - {exc}")
            raise
