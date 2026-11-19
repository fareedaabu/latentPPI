#!/bin/bash
# Resolve the helper relative to this script, regardless of the caller's directory.
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
exec "${PYTHON_BIN:-python}" "$SCRIPT_DIR/latentppi.py" clean "$@"
