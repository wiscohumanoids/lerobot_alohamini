#!/bin/bash
# Join script for attaching to a running AlohaMini Docker container

set -e  # Exit on error

# Configuration (must match run.sh)
CONTAINER_NAME="alohamini-dev"

echo "========================================"
echo "Joining AlohaMini Docker Container"
echo "========================================"
echo "Container: ${CONTAINER_NAME}"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker is not installed or not in PATH"
    exit 1
fi

# Check if container is running
if ! docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "ERROR: Container '${CONTAINER_NAME}' is not running"
    echo ""
    echo "To start the container, run:"
    echo "  ./run.sh"
    echo ""
    echo "Running containers:"
    docker ps --format "  - {{.Names}} ({{.Status}})"
    exit 1
fi

echo "Attaching to container..."
echo "Type 'exit' to detach (container will keep running)"
echo ""

# Execute a new bash shell in the running container
docker exec -it "${CONTAINER_NAME}" /bin/bash
