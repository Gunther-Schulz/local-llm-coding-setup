#!/usr/bin/env bash
# Setup for runpod: conda vLLM env, pip wheel cache, llama.cpp (CUDA).
# Usage: ./setup/install.sh
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WHEEL_CACHE="${WHEEL_CACHE:-$ROOT/.wheels}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$WHEEL_CACHE}"

echo "=== Runpod setup (vLLM env + llama.cpp CUDA) ==="
echo ""

# --- 1. Conda ---
if ! command -v conda &>/dev/null; then
  echo "ERROR: conda not found. Install miniconda or anaconda first."
  exit 1
fi
CONDA_BASE=$(conda info --base 2>/dev/null)
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [ ! -f "$CONDA_SH" ]; then
  echo "ERROR: conda.sh not at $CONDA_SH"
  exit 1
fi
. "$CONDA_SH"
echo "✓ Conda: $CONDA_BASE"
echo ""

# --- 2. vLLM env ---
if conda env list | grep -q "^vLLM "; then
  echo "✓ Conda env 'vLLM' already exists"
else
  echo "Creating conda env 'vLLM' (Python 3.10)..."
  conda create -n vLLM python=3.10 -y
  echo "✓ Created vLLM"
fi
conda activate vLLM
echo ""

# --- 3. Pip cache and upgrade ---
mkdir -p "$WHEEL_CACHE"
echo "Pip cache: $PIP_CACHE_DIR"
pip install -U pip
if [ -f "$ROOT/requirements.txt" ]; then
  echo "Installing from requirements.txt..."
  pip install -r "$ROOT/requirements.txt"
  echo "✓ Dependencies installed"
else
  echo "  (no requirements.txt; skip pip install)"
fi
echo ""

# --- 4. llama.cpp CUDA ---
echo "=== Build llama.cpp (CUDA) ==="
"$ROOT/setup/build/llamacpp_cuda.sh"
echo ""

echo "========================================="
echo "✓ Setup complete!"
echo ""
echo "Environment: vLLM"
echo "  Activate: conda activate vLLM"
echo "Wheel cache: $PIP_CACHE_DIR"
echo "llama-server: $ROOT/external/llama.cpp/build-cuda/bin/llama-server"
echo ""
echo "Next: ./run_server.sh [PORT]"
echo "  Config: config/server.env + config/models/<ACTIVE_MODEL>.yaml"
echo ""
echo "Update llama.cpp: ./setup/build/update_llamacpp.sh"
echo "========================================="
