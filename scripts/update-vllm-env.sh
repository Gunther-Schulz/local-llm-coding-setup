#!/usr/bin/env bash
# Update the vLLM conda env: pip/setuptools/wheel, transformers (for glm4_moe_lite etc.), optionally vllm.
# With --all: also upgrade every outdated pip package (may break torch/CUDA compatibility).
# Usage: ./scripts/update-vllm-env.sh [--all]
set -e

CONDA_ENV="${VLLM_CONDA_ENV:-vLLM}"

if ! command -v conda &>/dev/null; then
  echo "conda not found. Install Miniconda/Anaconda." >&2
  exit 1
fi

if [[ "${1:-}" == "--all" ]]; then
  UPGRADE_ALL=1
else
  UPGRADE_ALL=0
fi

echo "Updating conda env: $CONDA_ENV"

# Conda packages in this env (if any)
conda run -n "$CONDA_ENV" conda update -n "$CONDA_ENV" --all -y || true

# Pip base
conda run -n "$CONDA_ENV" pip install --upgrade pip setuptools wheel

# Upgrade vllm first. vLLM 0.15.x pins transformers<5; if we upgrade both in one go, pip keeps transformers 4.x.
conda run -n "$CONDA_ENV" pip install --upgrade vllm

# Then force transformers to 5.x (for glm4_moe_lite / GLM-4.7-Flash-MXFP4). Pip may warn about vllm conflict; ignore.
conda run -n "$CONDA_ENV" pip install --upgrade "transformers>=5.0"

if [[ "$UPGRADE_ALL" -eq 1 ]]; then
  echo "Upgrading all outdated pip packages (may break torch/CUDA)..."
  outdated=$(conda run -n "$CONDA_ENV" pip list --outdated --format=json 2>/dev/null | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    print(' '.join(x['name'] for x in data))
except Exception:
    pass
" || true)
  if [[ -n "$outdated" ]]; then
    conda run -n "$CONDA_ENV" pip install --upgrade $outdated
  else
    echo "No outdated pip packages."
  fi
else
  echo "Done. Run with --all to upgrade every outdated pip package (use with caution)."
fi

echo "Env $CONDA_ENV updated."
