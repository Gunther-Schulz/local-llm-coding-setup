#!/bin/bash
# Clean install for vLLM on RTX 5090 (local). Run from project root.
#
# Prerequisites: conda
# Usage: ./setup/install.sh
#
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Local wheel cache (persists across -r, not in git)
WHEEL_CACHE="$ROOT/.wheels"
mkdir -p "$WHEEL_CACHE"

echo "=== vLLM Setup for RTX 5090 ==="
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

# --- 2. Remove old 'llm' env, create fresh 'vLLM' env ---
echo "=== Clean slate: remove old envs, create 'vLLM' ==="
conda deactivate 2>/dev/null || true
conda env remove -n llm -y 2>/dev/null || echo "  (no 'llm' env to remove)"
conda env remove -n vLLM -y 2>/dev/null || echo "  (no 'vLLM' env to remove)"
conda create -n vLLM python=3.10 -y
conda activate vLLM
echo "✓ Environment: vLLM (Python 3.10)"
echo ""

# --- 3. PyTorch 2.9.1 cu128 (matches prebuilt vLLM wheel CUDA version) ---
echo "=== PyTorch 2.9.1 (CUDA 12.8 - matches vLLM prebuilt wheel) ==="
if [ ! -f "$WHEEL_CACHE/torch"*"2.9.1"*"cu128"*.whl ]; then
  echo "Downloading PyTorch wheels to $WHEEL_CACHE..."
  pip download torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --dest "$WHEEL_CACHE" || true
fi
if pip install --no-index --find-links="$WHEEL_CACHE" \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 2>/dev/null; then
  echo "✓ PyTorch installed from cache"
else
  echo "No matching cached wheel; installing PyTorch from index..."
  pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128
  echo "✓ PyTorch installed"
fi
echo ""

# --- 4. vLLM 0.14.1 (prebuilt, or build from source if no matching wheel) ---
echo "=== vLLM 0.14.1 (prebuilt with RTX 5090 support) ==="
if [ ! -f "$WHEEL_CACHE/vllm-0.14.1"*.whl ] 2>/dev/null; then
  echo "Downloading vLLM 0.14.1 wheel to $WHEEL_CACHE..."
  pip download vllm==0.14.1 \
    --dest "$WHEEL_CACHE" \
    --extra-index-url https://download.pytorch.org/whl/cu128 || true
fi
if pip install --no-index --find-links="$WHEEL_CACHE" vllm==0.14.1 2>/dev/null; then
  echo "✓ vLLM 0.14.1 installed from cache"
else
  echo "No matching prebuilt wheel; building vLLM 0.14.1 from source (this may take a while)..."
  pip install vllm==0.14.1 --no-binary vllm
  echo "✓ vLLM 0.14.1 built and installed"
fi
echo ""

# --- 5. Verify ---
echo "=== Verification ==="
python -c "import torch; print(f'✓ torch {torch.__version__}')"
python -c "import vllm; print(f'✓ vllm {vllm.__version__}')"
python -c "from vllm.entrypoints.cli.main import main; print('✓ vllm CLI imported')"
echo ""

# --- 6. Project deps ---
echo "=== Project dependencies (stack/, run/) ==="
pip install -r "$ROOT/requirements.txt"
echo "✓ Project deps installed"
echo ""

# --- 7. Build llama.cpp (vision + CUDA) ---
echo "=== Build llama.cpp (vision + CUDA) ==="
"$ROOT/setup/build/llamacpp_vision.sh"
"$ROOT/setup/build/llamacpp_cuda.sh"
echo "✓ llama.cpp built (vision: external/llama.cpp/build/, CUDA: external/llama.cpp/build-cuda/)"
echo ""

echo "========================================="
echo "✓ Setup complete!"
echo ""
echo "Environment: vLLM"
echo "To activate: conda activate vLLM"
echo ""
echo "Installation summary:"
echo "  • PyTorch 2.9.1+cu128 (CUDA 12.8) – from cache or index"
echo "  • vLLM 0.14.1 – from cache or built from source if no matching wheel"
echo "  • Python dependencies from requirements.txt"
echo "  • llama.cpp (vision CPU + CUDA for LLM/benchmark)"
echo "  • Re-runs: <1 min if wheels cached; vLLM build-from-source ~10–30 min if needed)"
echo ""
echo "Next steps:"
echo "  1. Select model:   ./run/run select model"
echo "  2. Start LLM:      ./run/run llm   (vLLM or llamacpp from config)"
echo "  3. Start proxy:    ./run/run proxy"
echo "  • Engine:         ./run/run select engine   (vllm | llamacpp)"
echo ""
echo "Optional:"
echo "  • Vision:   ./stack/download_vision_model.sh qwen2-vl-2b-q4, then ./run/run vision"
echo "  • llamacpp: ./run/run select engine llamacpp, then ./run/run llm"
echo "  • Benchmark: ./benchmark/run_benchmark.sh"
echo "========================================="
