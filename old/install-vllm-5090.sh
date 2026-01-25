#!/bin/bash
# Install everything for vLLM on RTX 5090: env, torch (cu128), deps, vLLM from source (sm_100).
# Run from project root. Needs: conda, CUDA 12.8+, nvcc, cmake. vllm/ must exist.
#
#   ./install-vllm-5090.sh       # fresh or reinstall
#   ./install-vllm-5090.sh -r    # nuke env first, then install (clean slate)
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

# Find conda
_conda_sh=""
for d in "$ROOT/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.miniconda3" "/workspace/miniconda3"; do
  [ -f "${d}/etc/profile.d/conda.sh" ] && { _conda_sh="${d}/etc/profile.d/conda.sh"; break; }
done
[ -z "$_conda_sh" ] && command -v conda &>/dev/null && {
  _b=$(conda info --base 2>/dev/null)
  [ -n "$_b" ] && [ -f "${_b}/etc/profile.d/conda.sh" ] && _conda_sh="${_b}/etc/profile.d/conda.sh"
}
[ -z "$_conda_sh" ] && { echo "ERROR: conda not found"; exit 1; }
. "$_conda_sh"

if [[ "$1" == "-r" || "$1" == "--reset" ]]; then
  echo "=== Remove env 'llm' ==="
  conda deactivate 2>/dev/null || true
  conda env remove -n llm -y 2>/dev/null || true
  echo ""
fi

# Create env if missing
if ! conda env list 2>/dev/null | grep -qw llm; then
  echo "=== Create env 'llm' (Python 3.10) ==="
  conda create -n llm python=3.10 -y
  echo ""
fi
conda activate llm

echo "=== 1. PyTorch (CUDA 12.8) ==="
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

echo ""
echo "=== 2. Deps (requirements.txt minus vllm) ==="
pip install -r <(grep -v '^#' "$ROOT/requirements.txt" | grep -v '^[[:space:]]*$' | grep -v '^vllm')

echo ""
echo "=== 3. Update vllm (git pull main) ==="
VLLM_SRC="${VLLM_SRC:-$ROOT/vllm}"
if [ -d "$VLLM_SRC/.git" ] || [ -f "$VLLM_SRC/.git" ]; then
  ( cd "$VLLM_SRC" && git fetch origin && git checkout main && git pull origin main )
else
  echo "  (skip: not a git repo)"
fi

echo ""
echo "=== 4. Build vLLM from source (sm_100 for 5090) ==="
[ ! -f "$VLLM_SRC/setup.py" ] && { echo "ERROR: vllm not at $VLLM_SRC"; exit 1; }
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.9 9.0 10.0}"
pip install -e "$VLLM_SRC"
pip install -U sse-starlette

echo ""
echo "Done. Run: ./start-vllm-server.sh"
