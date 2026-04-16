#!/bin/bash

# Binarization script using kraken
# Usage: ./binarization.sh <image_path> [output_dir]

INPUT_PATH="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RUNTIME_ROOT="${SMARTMATCH_RUNTIME_DIR:-$SCRIPT_DIR/.runtime}"
LOG_DIR="$RUNTIME_ROOT/logs"
LOG_FILE="$LOG_DIR/binarization.log"
OUTPUT_DIR="${2:-$RUNTIME_ROOT/binarized}"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

INPUT_FILENAME="$(basename "$INPUT_PATH")"
OUTPUT_PATH="$OUTPUT_DIR/$INPUT_FILENAME"

echo "[$(date)] Input: $INPUT_PATH -> $OUTPUT_PATH" >> "$LOG_FILE"

if kraken -i "$INPUT_PATH" "$OUTPUT_PATH" binarize; then
    echo "[$(date)] Status: SUCCESS" >> "$LOG_FILE"
    echo "Binarization completed successfully. Output saved to $OUTPUT_PATH"
    exit 0
else
    echo "[$(date)] Status: FAILED" >> "$LOG_FILE"
    echo "Binarization failed. Check the input image path and ensure kraken is installed."
    exit 1
fi
