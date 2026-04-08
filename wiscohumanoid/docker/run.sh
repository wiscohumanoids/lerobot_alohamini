#!/bin/bash
# Run script for AlohaMini Docker container

set -e  # Exit on error

# Detect script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"  # Go up to lerobot_alohamini/

# Configuration
IMAGE_NAME="alohamini"
IMAGE_TAG="latest"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
CONTAINER_NAME="alohamini-dev"

# Parse command line arguments
MODE="interactive"
if [ "$1" == "--jupyter" ]; then
    MODE="jupyter"
fi

echo "========================================"
echo "Running AlohaMini Docker Container"
echo "========================================"
echo "Image: ${FULL_IMAGE_NAME}"
echo "Mode: ${MODE}"
echo "Project root: ${PROJECT_ROOT}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

# Check if image exists
if ! docker image inspect "${FULL_IMAGE_NAME}" > /dev/null 2>&1; then
    echo "ERROR: Docker image '${FULL_IMAGE_NAME}' not found"
    echo "Please build the image first: docker/build.sh"
    exit 1
fi

# Stop and remove existing container if it exists
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "Stopping existing container..."
    docker stop "${CONTAINER_NAME}" > /dev/null 2>&1 || true
    docker rm "${CONTAINER_NAME}" > /dev/null 2>&1 || true
fi

XSOCK=/tmp/.X11-unix
XAUTH=/tmp/.docker.xauth

# Common Docker run arguments
DOCKER_ARGS=(
    --name "${CONTAINER_NAME}"
    --rm
    -it
    -v "${PROJECT_ROOT}:/workspace"
    -w /workspace
    -p 8888:8888
    -p 5555:5555
    -p 5556:5556
    -v "/tmp/.X11-unix:/tmp/.X11-unix"
    -e "DISPLAY=${DISPLAY}"
    -e "PYTHONPATH=/workspace/src"
    -v "$XSOCK:$XSOCK"
    -v "$XAUTH:$XAUTH"
    -e "XAUTHORITY=$XAUTH"
    -e "XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR"
    -v "$XDG_RUNTIME_DIR:$XDG_RUNTIME_DIR"
    --privileged
    -v /dev/bus/usb:/dev/bus/usb
    -v "${SCRIPT_DIR}/../checkpoints:/root/.cache/huggingface/"
    -v "${SCRIPT_DIR}/../calibration/leader:/root/.cache/huggingface/lerobot/calibration/teleoperators/so_leader"
    -v "${SCRIPT_DIR}/../calibration/robot:/root/.cache/huggingface/lerobot/calibration/robots/lekiwi"
    -v "${SCRIPT_DIR}/../calibration/robot:/root/.cache/huggingface/lerobot/calibration/robots/lekiwi_client"
)

if command -v nvidia-smi &> /dev/null && nvidia-smi -L &> /dev/null; then
    DOCKER_ARGS+=(
        --runtime=nvidia
        --gpus all
        -e NVIDIA_DRIVER_CAPABILITIES=all
    )
fi

if [[ "$OSTYPE" == "darwin"* || "$OSTYPE" == "linux-gnu"* ]]; then
    DOCKER_ARGS+=(
        --pid=host
    )
fi

# Add device passthrough if devices exist (for hardware deployment)
if [ -d "/dev" ]; then
    # USB serial devices (motor controllers)
    if ls /dev/ttyUSB* 1> /dev/null 2>&1; then
        echo "Found USB serial devices, adding to container..."
        for device in /dev/ttyUSB*; do
            DOCKER_ARGS+=(--device="${device}")
        done
    fi

    # USB video devices (cameras)
    if ls /dev/video* 1> /dev/null 2>&1; then
        echo "Found video devices, adding to container..."
        for device in /dev/video*; do
            DOCKER_ARGS+=(--device="${device}")
        done
    fi

    # Stable by-path symlinks (created automatically by Linux for V4L2 devices)
    if [ -d /dev/v4l/by-path ]; then
        echo "Mounting stable V4L2 by-path symlinks..."
        DOCKER_ARGS+=(-v /dev/v4l:/dev/v4l)
    fi

    # Audio devices (for voice control)
    if [ -e "/dev/snd" ]; then
        DOCKER_ARGS+=(--device=/dev/snd)
    fi
fi


# Run container based on mode
if [ "${MODE}" == "jupyter" ]; then
    echo ""
    echo "Starting Jupyter Lab..."
    echo "Access at: http://localhost:8888"
    echo ""
    echo "Press Ctrl+C to stop"
    echo ""

    docker run "${DOCKER_ARGS[@]}" \
        "${FULL_IMAGE_NAME}" \
        jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''
else
    echo ""
    echo "Starting interactive bash shell..."
    echo ""
    echo "Available commands:"
    echo "  - View documentation: cat docs/ALOHAMINI_ARCHITECTURE.md"
    echo "  - Run notebook: jupyter notebook docs/AlohaMini_Walkthrough.ipynb"
    echo "  - Read capabilities: cat docs/ALOHAMINI_CAPABILITIES_REPORT.md"
    echo "  - View examples: ls workspace/examples/"
    echo "  - Find cameras: python -m lerobot.scripts.lerobot_find_cameras"
    echo "  - Exit container: exit"
    echo ""

    docker run "${DOCKER_ARGS[@]}" \
        "${FULL_IMAGE_NAME}" \
        /bin/bash
fi
