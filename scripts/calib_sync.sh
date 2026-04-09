#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ROBOT_DIR="$(realpath "$SCRIPT_DIR/../calibration/robot")"
mkdir -p "$ROBOT_DIR"
if [ -d "$ROBOT_DIR" ]; then
    cp -av "$ROBOT_DIR/." "/root/.cache/huggingface/lerobot/calibration/robots/lekiwi_client/"
    cp -av "$ROBOT_DIR/." "/root/.cache/huggingface/lerobot/calibration/robots/lekiwi/"
else
    echo "Error: Source directory $ROBOT_DIR does not exist."
    exit 1
fi

LEADER_DIR="$(realpath "$SCRIPT_DIR/../calibration/leader")"
mkdir -p "$LEADER_DIR"
if [ -d "$LEADER_DIR" ]; then
    cp -av "$LEADER_DIR/." "/root/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/"
else
    echo "Error: Source directory $LEADER_DIR does not exist."
    exit 1
fi