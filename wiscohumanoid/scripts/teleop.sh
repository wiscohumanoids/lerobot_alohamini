#!/bin/bash

# Resolve the directory of the shell script to ensure relative path stability
SCRIPT_DIR=$(dirname "$(readlink -f "$0")")

# Execute Python script with passed arguments
python3 "$SCRIPT_DIR/../../src/utils/teleoperate.py" "$@"