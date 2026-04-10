#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_CONFIG_DIR="$(realpath "$SCRIPT_DIR/../calibration/")"
HF_CONFIG_DIR="$HOME/.cache/huggingface/lerobot/"

mkdir -p "$HF_CONFIG_DIR"
if [ -d "$HF_CONFIG_DIR" ]; then
    cp -r "$SRC_CONFIG_DIR" "$HF_CONFIG_DIR/"
else
    echo "Error copying configs!"
    exit 1
fi