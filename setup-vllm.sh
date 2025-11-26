#!/bin/bash
set -e

echo "Setting up vLLM in the RunPod environment (system Python)..."

cd /workspace

echo "Installing / upgrading vLLM..."
python3 -m pip install -U vllm sse-starlette

echo ""
echo "✅ vLLM setup complete."
echo ""
echo "You can now start the vLLM OpenAI-compatible server with:"
echo "  ./start-vllm-server.sh"


