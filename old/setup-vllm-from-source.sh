#!/bin/bash
# Build vLLM from source with sm_100 (10.0) for RTX 5090.
# Use this instead of setup-vllm.sh if you need native 5090 kernels; the
# prebuilt PyPI wheel does not include 10.0.
#
# Requires: CUDA 12.8+, nvcc, conda/venv with PyTorch 2.7+ (cu128). The vllm/
# subdir must exist (git submodule or clone).
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

_conda_sh=""
for d in "$ROOT/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.miniconda3" "/workspace/miniconda3"; do
  if [ -f "${d}/etc/profile.d/conda.sh" ]; then
    _conda_sh="${d}/etc/profile.d/conda.sh"
    break
  fi
done
[ -z "$_conda_sh" ] && command -v conda &>/dev/null && {
  _base=$(conda info --base 2>/dev/null)
  [ -n "$_base" ] && [ -f "${_base}/etc/profile.d/conda.sh" ] && _conda_sh="${_base}/etc/profile.d/conda.sh"
}
if [ -n "$_conda_sh" ]; then
  . "$_conda_sh"
  conda activate llm
elif [ -n "$VIRTUAL_ENV" ]; then
  :
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  . "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  . "$ROOT/venv/bin/activate"
else
  echo "ERROR: No conda or venv. Create one and run ./install-vllm-5090.sh first."
  exit 1
fi

VLLM_SRC="${VLLM_SRC:-$ROOT/vllm}"
if [ ! -f "$VLLM_SRC/setup.py" ]; then
  echo "ERROR: vllm source not found at $VLLM_SRC (set VLLM_SRC to override)"
  exit 1
fi

# RTX 5090 = sm_100 = 10.0. Also include 8.0, 8.9, 9.0 for compatibility.
# Do NOT use 12.0/12.1 here — those are for B200/RTX 6000.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0 8.9 9.0 10.0}"

echo "Building vLLM from source for RTX 5090 (sm_100)..."
echo "  TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"
echo "  Source: $VLLM_SRC"
echo ""

pip install -e "$VLLM_SRC"
pip install -U sse-starlette

echo ""
echo "✅ vLLM built from source with 10.0 (sm_100)."
echo "   Start server: ./start-vllm-server.sh"
