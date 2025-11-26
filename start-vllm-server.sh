#!/bin/bash
set -e

# Start an OpenAI-compatible vLLM server using the existing GGUF model.
# This uses vLLM's experimental GGUF support so we can reuse the local
# Qwen2.5-Coder-14B Q4_K_M GGUF without downloading a new checkpoint.
# You can override via:
#   export VLLM_GGUF_MODEL="/path/to/model.gguf"
#   export VLLM_TOKENIZER_ID="Qwen/Qwen2.5-Coder-14B-Instruct"
#   export VLLM_MAX_LEN=81920

GGUF_MODEL="${VLLM_GGUF_MODEL:-/workspace/models/qwen2.5-coder-14b-q4_k_m/qwen2.5-coder-14b-instruct-q4_k_m.gguf}"
TOKENIZER_ID="${VLLM_TOKENIZER_ID:-Qwen/Qwen2.5-Coder-14B-Instruct}"
# vLLM with GGUF is limited to native training context (no RoPE scaling support)
MAX_LEN="${VLLM_MAX_LEN:-32768}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

cd /workspace

if [ -f "/workspace/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source /workspace/miniconda3/etc/profile.d/conda.sh
  conda activate glm4
else
  echo "WARNING: conda environment not found at /workspace/miniconda3; using system Python."
fi

echo "🚀 Starting vLLM OpenAI server..."
echo "  GGUF model     : $GGUF_MODEL"
echo "  Tokenizer      : $TOKENIZER_ID"
echo "  Max model len  : $MAX_LEN (native context - GGUF doesn't support RoPE scaling)"
echo "  Host / Port    : $HOST:$PORT"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model "$GGUF_MODEL" \
  --tokenizer "$TOKENIZER_ID" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype auto \
  --max-model-len "$MAX_LEN" \
  --tensor-parallel-size 1


