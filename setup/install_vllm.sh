#!/usr/bin/env bash
# Install vLLM into the project's conda env (vLLM). Use after ./setup/install.sh.
# Usage: ./setup/install_vllm.sh
# For a full fresh install: ./setup/install.sh  then  ./setup/install_vllm.sh
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WHEEL_CACHE="${WHEEL_CACHE:-$ROOT/.wheels}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WHEEL_CACHE}"

if ! command -v conda &>/dev/null; then
  echo "conda not found. Run ./setup/install.sh first." >&2
  exit 1
fi
CONDA_BASE=$(conda info --base 2>/dev/null)
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
[[ -f "$CONDA_SH" ]] || { echo "conda.sh not at $CONDA_SH" >&2; exit 1; }
. "$CONDA_SH"

if ! conda env list | grep -q "^vLLM "; then
  echo "Conda env 'vLLM' not found. Run ./setup/install.sh first to create it."
  exit 1
fi

echo "Installing vLLM into conda env 'vLLM'..."
mkdir -p "$WHEEL_CACHE"
conda run -n vLLM pip install -U pip
conda run -n vLLM pip install -U vllm
echo "Done. Verify with: conda run -n vLLM python -c 'import vllm; print(vllm.__version__)'"
echo "To use the vLLM backend: set a model with backend: vllm in config/models/<key>.yaml, then ./run_server.sh <key>"
