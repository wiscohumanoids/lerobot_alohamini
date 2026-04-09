#!/bin/bash

set -e

LEKIWI_CLIENT_DIR="/root/.cache/huggingface/lerobot/calibration/robots/lekiwi_client"
LEKIWI_DIR="/root/.cache/huggingface/lerobot/calibration/robots/lekiwi"
SO_DIR="/root/.cache/huggingface/lerobot/calibration/teleoperators/so_leader"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROBOT_DIR="$(realpath "$SCRIPT_DIR/../calibration/robot")"
LEADER_DIR="$(realpath "$SCRIPT_DIR/../calibration/leader")"

mkdir -p "$ROBOT_DIR"
mkdir -p "$LEKIWI_CLIENT_DIR"

if [ -d "$ROBOT_DIR" ]; then
    cp -av "$ROBOT_DIR/." "$LEKIWI_CLIENT_DIR/"
    cp -av "$ROBOT_DIR/." "$LEKIWI_DIR/"
else
    echo "Error: Source directory $ROBOT_DIR does not exist."
    exit 1
fi


mkdir -p "$LEADER_DIR"
mkdir -p "$SO_DIR"
if [ -d "$LEADER_DIR" ]; then
    cp -av "$LEADER_DIR/." "$SO_DIR/"
else
    echo "Error: Source directory $LEADER_DIR does not exist."
    exit 1
fi
