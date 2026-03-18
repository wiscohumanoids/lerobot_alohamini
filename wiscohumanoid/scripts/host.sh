#!/usr/bin/env bash

set -euo pipefail
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
#readonly LOG_FILE="/tmp/$(basename "$0").log"

log() {
    echo "[$(date +'%Y-%m-%d @ %H:%M:%S%z')] : $*" #| tee -a "$LOG_FILE"
}

VERBOSE=false
if [[ "$*" == *"-v"* ]] || [[ "$*" == *"--verbose"* ]]; then
  VERBOSE=true
fi

main() {
    if [ -f /.dockerenv ]; then
        log "Starting AlohaMini host environment (container)"
    else
        log "Failed to detect active Docker container in current setup, cannot start host env!"
        log "(Make sure to run ../docker/run.sh and execute this script INSIDE that container)"
        return -1
    fi

    if [[ "$VERBOSE" == true ]]; then
        log "VERBOSE output enabled"
    fi

    log "Server starting"
    #if [[ "$VERBOSE" == true ]]; then
    #    python -m lerobot.robots.alohamini.lekiwi_host
    #else
    #    python -m lerobot.robots.alohamini.lekiwi_host --verbose
    #fi

    python -m lerobot.robots.alohamini.lekiwi_host
}

main