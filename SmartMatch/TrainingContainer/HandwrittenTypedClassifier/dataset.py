import os
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torch


_config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(_config_path, "r") as f:
    _config = json.load(f)

TARGET_SIZE = tuple(_config["target_size"])  # (height, width)
TARGET_SIZE = (
    int(os.getenv("SMARTMATCH_CLASSIFIER_TARGET_HEIGHT", TARGET_SIZE[0])),
    int(os.getenv("SMARTMATCH_CLASSIFIER_TARGET_WIDTH", TARGET_SIZE[1])),
)


class SimpleImageDataset(Dataset):
    def __init__(self, file_paths, labels, target_size=TARGET_SIZE):
        self.file_paths = file_paths
        self.labels = labels
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img_path = self.file_paths[idx]
        # Convert to grayscale, resize with LANCZOS (high quality but fast enough)
        img = Image.open(img_path).convert('L')
        img = img.resize((self.target_size[1], self.target_size[0]), Image.Resampling.LANCZOS)
        img_np = np.array(img, dtype=np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).unsqueeze(0)   # (1, H, W)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return img_tensor, label
