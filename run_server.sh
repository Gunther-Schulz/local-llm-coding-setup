#!/usr/bin/env bash
# Start llama-server with Qwen3-Coder-Next MXFP4_MOE (OpenAI-compatible API).
# No dependency on benchmark/ — use benchmark/next/ for measuring only.
# Runs in project conda env (runpod or vLLM). Usage: ./run_server.sh [PORT]
#   PORT defaults to 8000. Override context: SERVER_CTX=32768 ./run_server.sh
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Activate project conda env (runpod or vLLM) so PATH/libs are correct
if [[ -f "$ROOT/old_setup/lib/activate.sh" ]]; then
  source "$ROOT/old_setup/lib/activate.sh"
fi

PORT="${1:-8000}"

echo "Model name for Cursor: qwen3-coder-next"
echo "API base URL: http://127.0.0.1:$PORT/v1"
echo ""

LLAMA_SERVER="${LLAMACPP_SERVER_BIN:-$ROOT/external/llama.cpp/build-cuda/bin/llama-server}"
if [[ ! -x "$LLAMA_SERVER" && "$LLAMA_SERVER" != /* ]]; then
  LLAMA_SERVER="$ROOT/$LLAMA_SERVER"
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found: $LLAMA_SERVER"
  echo "Set LLAMACPP_SERVER_BIN or build: ./old_setup/setup/build/llamacpp_cuda.sh"
  exit 1
fi

MODEL_DIR="$ROOT/models/qwen3-coder-next-mxfp4"
MODEL_FILE="$MODEL_DIR/Qwen3-Coder-Next-MXFP4_MOE.gguf"
if [[ ! -f "$MODEL_FILE" ]]; then
  if [[ -d "$MODEL_DIR" ]]; then
    MODEL_FILE=$(find "$MODEL_DIR" -maxdepth 1 -name "*.gguf" -print -quit)
  fi
fi
if [[ -z "$MODEL_FILE" || ! -f "$MODEL_FILE" ]]; then
  echo "Model not found: $MODEL_DIR/Qwen3-Coder-Next-MXFP4_MOE.gguf (or any .gguf in $MODEL_DIR)"
  echo "Download the MXFP4_MOE GGUF into $MODEL_DIR (e.g. via benchmark/next/download.sh or Hugging Face)."
  exit 1
fi

ctx="${SERVER_CTX:-262144}"
echo "port=$PORT ctx=$ctx n_gpu_layers=-1 model=$MODEL_FILE"
exec "$LLAMA_SERVER" -m "$MODEL_FILE" --host "127.0.0.1" --port "$PORT" --n-gpu-layers -1 --jinja -c "$ctx"
