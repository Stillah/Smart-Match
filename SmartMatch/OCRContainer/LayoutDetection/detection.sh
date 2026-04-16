#!/bin/bash

# Layout detection script
# Usage: ./detection.sh <image_path> <output_dir>

INPUT_PATH="$1"
OUTPUT_DIR="${2:?Error: output directory is required}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
CONTAINER_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RUNTIME_ROOT="${SMARTMATCH_RUNTIME_DIR:-$CONTAINER_ROOT/.runtime}"
LOG_DIR="$RUNTIME_ROOT/logs"
LOG_FILE="$LOG_DIR/detection.log"

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"

echo "[$(date)] Input: $INPUT_PATH, Output dir: $OUTPUT_DIR" >> "$LOG_FILE"

if python "$SCRIPT_DIR/detection.py" "$INPUT_PATH" "$OUTPUT_DIR" >> "$LOG_FILE" 2>&1; then
    echo "[$(date)] Status: SUCCESS" >> "$LOG_FILE"
    echo "Detection completed successfully. Segments saved to $OUTPUT_DIR"
    exit 0
else
    echo "[$(date)] Status: FAILED" >> "$LOG_FILE"
    echo "Detection failed. Check the log at $LOG_FILE for details."
    exit 1
fi
