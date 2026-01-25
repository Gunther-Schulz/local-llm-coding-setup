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
# Download to local cache first (only fetches if missing)
if [ ! -f "$WHEEL_CACHE/torch"*"2.9.1"*"cu128"*.whl ]; then
  echo "Downloading PyTorch wheels to $WHEEL_CACHE..."
  pip download torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
    --index-url https://download.pytorch.org/whl/cu128 \
    --dest "$WHEEL_CACHE"
else
  echo "Using cached PyTorch wheels from $WHEEL_CACHE"
fi
# Install from local cache (fast)
pip install --no-index --find-links="$WHEEL_CACHE" \
  torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1
echo "✓ PyTorch installed"
echo ""

# --- 4. vLLM 0.14.1 (prebuilt, with sm_100 for RTX 5090) ---
echo "=== vLLM 0.14.1 (prebuilt with RTX 5090 support) ==="
# Download to local cache first (only fetches if missing)
if [ ! -f "$WHEEL_CACHE/vllm-0.14.1"*.whl ]; then
  echo "Downloading vLLM 0.14.1 wheel to $WHEEL_CACHE..."
  pip download vllm==0.14.1 \
    --dest "$WHEEL_CACHE" \
    --extra-index-url https://download.pytorch.org/whl/cu128
else
  echo "Using cached vLLM wheel from $WHEEL_CACHE"
fi
# Install from local cache (fast)
pip install --no-index --find-links="$WHEEL_CACHE" vllm==0.14.1
echo "✓ vLLM 0.14.1 installed"
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

echo "========================================="
echo "✓ vLLM setup complete!"
echo ""
echo "Environment: vLLM"
echo "To activate: conda activate vLLM"
echo ""
echo "Installation summary:"
echo "  • PyTorch 2.9.1+cu128 (CUDA 12.8)"
echo "  • vLLM 0.14.1 (prebuilt, RTX 5090 sm_100 support)"
echo "  • Install time: 2-5 minutes (NO building!)"
echo "  • Re-runs with -r: <1 minute (all cached)"
echo ""
echo "Next steps:"
echo "  1. Select a model: ./run/select_model.sh"
echo "  2. Start server:   ./run/vllm.sh"
echo "========================================="
