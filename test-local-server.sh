#!/bin/bash
# Test script for local vLLM server

echo "🚀 Starting vLLM server for local testing..."
echo "Model: Qwen2.5-Coder-7B"
echo "Context: 8K tokens"
echo "GPU Memory: 85% utilization"
echo ""

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate vllm-test

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-coder-7b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1

