#!/bin/bash
# Proper signal handling for graceful shutdown on CTRL-C

ROOT="${ROOT:-${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}}"
export ROOT
. "$ROOT/lib/activate.sh"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"

# Use exec to replace shell with Python process
# This ensures signals (CTRL-C) go directly to Python for proper cleanup
exec python -m run.vllm "$@"
