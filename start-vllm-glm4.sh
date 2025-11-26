#!/bin/bash
# Start vLLM server for GLM-4-32B-0414 AWQ with 64K context
# Run this on the RunPod instance

set -e

MODEL_DIR="/workspace/models/glm-4-32b-0414-awq"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.9}"

echo "🚀 Starting vLLM Server for GLM-4-32B-0414 AWQ"
echo "Model Directory: $MODEL_DIR"
echo "Port: $PORT"
echo "Max Context: $MAX_MODEL_LEN (64K with YaRN)"
echo "GPU Memory Utilization: $GPU_MEMORY_UTIL"
echo ""

# Check if model exists
if [ ! -d "$MODEL_DIR" ] || [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ Model not found at $MODEL_DIR"
    echo "Please download the model first: bash /workspace/scripts/download-glm4.sh"
    exit 1
fi

# Activate conda environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate vllm-test

# Check GPU
echo "📊 GPU Status:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader

# Allow longer context (for 64K+)
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_CACHE_ROOT=/workspace/.cache

# Start vLLM server
echo ""
echo "🔥 Starting vLLM server..."
echo "Server will be available at: http://0.0.0.0:$PORT"
echo "Press Ctrl+C to stop"
echo ""

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL_DIR" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --quantization awq \
    --tensor-parallel-size 1


