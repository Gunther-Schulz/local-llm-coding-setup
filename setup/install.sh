#!/bin/bash
# Clean install for runpod (llama-server + proxy). Run from project root.
#
# Prerequisites: conda
# Usage: ./setup/install.sh
#
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "=== Runpod setup (llama-server + proxy) ==="
echo ""

# --- 1. Find conda ---
if command -v conda &>/dev/null; then
  CONDA_BASE=$(conda info --base 2>/dev/null)
  CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
  if [ ! -f "$CONDA_SH" ]; then
    echo "ERROR: conda found but conda.sh not at $CONDA_SH"
    exit 1
  fi
else
  echo "ERROR: conda not found. Install miniconda or anaconda first."
  exit 1
fi

. "$CONDA_SH"
echo "✓ Conda: $CONDA_BASE"
echo ""

# --- 2. Remove old envs, create 'runpod' env ---
echo "=== Clean slate: remove old envs, create 'runpod' ==="
conda deactivate 2>/dev/null || true
conda env remove -n llm -y 2>/dev/null || echo "  (no 'llm' env to remove)"
conda env remove -n vLLM -y 2>/dev/null || echo "  (no 'vLLM' env to remove)"
conda env remove -n runpod -y 2>/dev/null || echo "  (no 'runpod' env to remove)"
conda create -n runpod python=3.10 -y
conda activate runpod
echo "✓ Environment: runpod (Python 3.10)"
echo ""

# --- 3. Project deps ---
echo "=== Project dependencies (stack/, run/, proxy) ==="
pip install -r "$ROOT/requirements.txt"
echo "✓ Project deps installed"
echo ""

# --- 4. Build llama.cpp (vision + CUDA) ---
echo "=== Build llama.cpp (vision + CUDA) ==="
"$ROOT/setup/build/llamacpp_vision.sh"
"$ROOT/setup/build/llamacpp_cuda.sh"
echo "✓ llama.cpp built (vision: external/llama.cpp/build/, CUDA: external/llama.cpp/build-cuda/)"
echo ""

echo "========================================="
echo "✓ Setup complete!"
echo ""
echo "Environment: runpod"
echo "To activate: conda activate runpod"
echo ""
echo "Installation summary:"
echo "  • Python dependencies from requirements.txt"
echo "  • llama.cpp (vision CPU + CUDA for LLM/benchmark)"
echo ""
echo "Next steps:"
echo "  1. Select model:   ./run/run select model"
echo "  2. Start LLM:      ./run/run llm   (llama-server)"
echo "  3. Start proxy:    ./run/run proxy"
echo ""
echo "Optional:"
echo "  • Vision:   ./stack/download_vision_model.sh qwen2-vl-2b-q4, then ./run/run vision"
echo "  • Benchmark: ./benchmark/run_benchmark.sh"
echo "========================================="
