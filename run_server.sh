#!/usr/bin/env bash
# Start llama-server. Config: config/server.env (active model) + config/models/<key>.env (per-model options).
# Usage: ./run_server.sh [PORT]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

# Load active model and server options
if [[ ! -f "$ROOT/config/server.env" ]]; then
  echo "Config not found: config/server.env" >&2
  exit 1
fi
set -a
source "$ROOT/config/server.env"
set +a
if [[ -z "$ACTIVE_MODEL" ]]; then
  echo "ACTIVE_MODEL not set in config/server.env" >&2
  exit 1
fi
# Port: argv overrides config
[[ -n "$1" ]] && PORT="$1"

# Load per-model options
MODEL_ENV="$ROOT/config/models/${ACTIVE_MODEL}.env"
if [[ ! -f "$MODEL_ENV" ]]; then
  echo "Model config not found: $MODEL_ENV" >&2
  exit 1
fi
set -a
source "$MODEL_ENV"
set +a

# Resolve model path
MODEL_PATH="$ROOT/models/${ACTIVE_MODEL}/${GGUF}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi

# Binary
LLAMA_SERVER="${LLAMACPP_SERVER_BIN:-$ROOT/external/llama.cpp/build-cuda/bin/llama-server}"
if [[ "$LLAMA_SERVER" != /* ]]; then
  LLAMA_SERVER="$ROOT/$LLAMA_SERVER"
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found: $LLAMA_SERVER" >&2
  echo "Run: ./setup/install.sh  or  ./setup/build/llamacpp_cuda.sh" >&2
  exit 1
fi

# Build argv (host, port, n_gpu_layers, jinja from config)
argv=(-m "$MODEL_PATH" --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" --n-gpu-layers "${N_GPU_LAYERS:--1}" -c "${CONTEXT_SIZE:-262144}")
[[ "${JINJA:-1}" =~ ^(1|true|on|yes)$ ]] && argv+=(--jinja)
[[ -n "$TEMP" ]]    && argv+=(--temp "$TEMP")
[[ -n "$TOP_P" ]]   && argv+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]]   && argv+=(--top-k "$TOP_K")
[[ -n "$MIN_P" ]]   && argv+=(--min-p "$MIN_P")
[[ -n "$SEED" ]]    && argv+=(--seed "$SEED")
[[ -n "$BATCH_SIZE" ]]  && argv+=(--batch-size "$BATCH_SIZE")
[[ -n "$UBATCH_SIZE" ]] && argv+=(--ubatch-size "$UBATCH_SIZE")

echo "port=${PORT:-8000} ctx=${CONTEXT_SIZE:-262144} model=$(basename "$MODEL_PATH")"
echo "API: http://${HOST:-127.0.0.1}:${PORT:-8000}/v1"
exec "$LLAMA_SERVER" "${argv[@]}"
