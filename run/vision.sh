#!/bin/bash

ROOT="${ROOT:-${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}}}"
export ROOT
. "$ROOT/lib/activate.sh"
cd "$ROOT"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
exec python -m run.vision "$@"
