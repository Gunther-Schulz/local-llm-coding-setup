#!/bin/bash
set -e

echo "Setting up vLLM in the RunPod environment..."

cd /workspace

# Activate existing conda env (glm4) if available
if [ -f "/workspace/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate glm4
else
  echo "WARNING: conda environment not found at /workspace/miniconda3; using system Python."
fi

echo "Installing / upgrading vLLM..."
pip install -U "vllm[all]" sse-starlette

echo ""
echo "✅ vLLM setup complete."
echo ""
echo "You can now start the vLLM OpenAI-compatible server with:"
echo "  ./start-vllm-server.sh"


