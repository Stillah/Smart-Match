import os
import sys
import glob
import json
import logging
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from model import SimpleCNN
from dataset import SimpleImageDataset

# ==================== CONFIGURATION ====================
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)


def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Handwritten vs Typed Classifier")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config.json (default: config.json beside this script)")
    parser.add_argument("--img_folder", type=str, default=None,
                        help="Path to folder containing training images (overrides config)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Path to output directory for weights and logs (overrides config)")
    return parser.parse_args()


def get_paths(args):
    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.config:
        config_path = args.config
    else:
        config_path = os.path.join(script_dir, "config.json")

    config = load_config(config_path)

    img_folder = args.img_folder if args.img_folder else config["img_folder"]
    if not os.path.isabs(img_folder):
        img_folder = os.path.normpath(os.path.join(script_dir, img_folder))

    output_dir = args.output_dir if args.output_dir else config["output_dir"]
    if not os.path.isabs(output_dir):
        output_dir = os.path.normpath(os.path.join(script_dir, output_dir))

    os.makedirs(output_dir, exist_ok=True)

    log_file = os.path.join(output_dir, "training.log")
    checkpoint_path = os.path.join(output_dir, "best_model.pth")

    return config, img_folder, output_dir, log_file, checkpoint_path


# ==================== DATA PREPARATION ====================
def prepare_data(img_folder, config, logger):
    png_files = glob.glob(os.path.join(img_folder, "*.png"))
    jpeg_files = (
        glob.glob(os.path.join(img_folder, "*.jpg"))
        + glob.glob(os.path.join(img_folder, "*.jpeg"))
    )

    logger.info(f"Found {len(png_files)} PNG images (class 1)")
    logger.info(f"Found {len(jpeg_files)} JPEG images (class 0)")

    file_paths = png_files + jpeg_files
    labels = [1] * len(png_files) + [0] * len(jpeg_files)

    X_train, X_temp, y_train, y_temp = train_test_split(
        file_paths, labels, test_size=config["test_split"], stratify=labels, random_state=config["random_state"]
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=config["val_split"], stratify=y_temp, random_state=config["random_state"]
    )

    logger.info(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    train_dataset = SimpleImageDataset(X_train, y_train)
    val_dataset = SimpleImageDataset(X_val, y_val)
    test_dataset = SimpleImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset


# ==================== TRAINING ====================
def train(model, train_loader, val_loader, train_dataset, val_dataset, config, logger, device, checkpoint_path):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(config["num_epochs"]):
        # ----- Training -----
        model.train()
        train_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Train]", leave=False)
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * inputs.size(0)
        train_loss /= len(train_dataset)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['num_epochs']} [Val]", leave=False)
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = torch.sigmoid(outputs) > 0.5
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        val_loss /= len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds)

        logger.info(
            f"Epoch {epoch+1}/{config['num_epochs']} | "
            f"Train Loss = {train_loss:.4f}, "
            f"Val Loss = {val_loss:.4f}, "
            f"Val F1 = {val_f1:.4f}"
        )

        # ----- Save best model & early stopping -----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)
            epochs_no_improve = 0
            logger.info("  --> New best model saved")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config["patience"]:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

    logger.info(f"Training finished. Best Val Loss: {best_val_loss:.4f}")


# ==================== EVALUATION ====================
def evaluate(model, test_loader, checkpoint_path, logger, device):
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    test_preds = []
    test_labels = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.sigmoid(outputs) > 0.5
            test_preds.extend(preds.cpu().numpy())
            test_labels.extend(labels.numpy())

    test_f1 = f1_score(test_labels, test_preds)
    accuracy = np.mean(np.array(test_preds) == np.array(test_labels))

    logger.info(f"Test F1 Score:  {test_f1:.4f}")
    logger.info(f"Test Accuracy: {accuracy:.4f}")


# ==================== MAIN ====================
if __name__ == "__main__":
    args = parse_args()
    config, img_folder, output_dir, log_file, checkpoint_path = get_paths(args)

    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Starting Handwritten vs Typed Classifier Training")
    logger.info(f"Image folder : {img_folder}")
    logger.info(f"Output dir  : {output_dir}")
    logger.info(f"Log file    : {log_file}")
    logger.info("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset = (
        prepare_data(img_folder, config, logger)
    )

    model = SimpleCNN().to(device)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    train(model, train_loader, val_loader, train_dataset, val_dataset, config, logger, device, checkpoint_path)

    evaluate(model, test_loader, checkpoint_path, logger, device)

    logger.info(f"Model weights saved to: {checkpoint_path}")
    logger.info(f"Training log saved to: {log_file}")
