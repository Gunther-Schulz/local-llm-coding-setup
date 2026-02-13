#!/usr/bin/env bash
# Core: start one llama-server. Used by run_chat.sh, run_coding.sh, run_notebook.sh.
# No default model — pass MODEL_KEY (and optional PORT). Use launchers for a mode.
# Config: config/server.env + config/models/<key>.yaml.
# Usage: ./run_server.sh [--verbose] MODEL_KEY [PORT]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

VERBOSE=""
ACTIVE_MODEL=""
PORT=""
for arg in "$@"; do
  if [[ "$arg" == "--verbose" ]]; then
    VERBOSE=1
  elif [[ -z "$ACTIVE_MODEL" ]]; then
    ACTIVE_MODEL="$arg"
  elif [[ -z "$PORT" && "$arg" =~ ^[0-9]+$ ]]; then
    PORT="$arg"
  fi
done
PORT="${PORT:-8001}"

if [[ -z "$ACTIVE_MODEL" ]]; then
  echo "Usage: ./run_server.sh [--verbose] MODEL_KEY [PORT]" >&2
  echo "  --verbose = pass --verbose to llama-server (e.g. for tool/template messages)" >&2
  echo "  MODEL_KEY = config/models/<key>.yaml (e.g. qwen3-coder-next-mxfp4)" >&2
  echo "  PORT      = optional, default 8001" >&2
  echo "Or run a mode: ./run_chat.sh  ./run_coding.sh  ./run_notebook.sh" >&2
  exit 1
fi

if [[ ! -f "$ROOT/config/server.env" ]]; then
  echo "Config not found: config/server.env" >&2
  exit 1
fi
set -a
source "$ROOT/config/server.env"
set +a

echo "Loading model: $ACTIVE_MODEL (stop any server on port ${PORT} first)"
echo "Use this name in Cursor: $ACTIVE_MODEL"

# Load per-model config (YAML -> env)
set -a
eval "$("$ROOT/scripts/load_model_config.sh" "$ACTIVE_MODEL")"
set +a
# server.env override for CPU threads (overrides YAML when set)
[[ -n "${LLAMA_THREADS:-}" ]] && THREADS="$LLAMA_THREADS"

# Resolve model path
MODEL_PATH="$ROOT/models/${ACTIVE_MODEL}/${GGUF}"
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH" >&2
  exit 1
fi
# Optional mmproj (vision model)
MMPROJ_PATH=""
if [[ -n "$MMPROJ" ]]; then
  MMPROJ_PATH="$ROOT/models/${ACTIVE_MODEL}/${MMPROJ}"
  if [[ ! -f "$MMPROJ_PATH" ]]; then
    echo "mmproj not found: $MMPROJ_PATH" >&2
    exit 1
  fi
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

# Log file (overwritten each run, no timestamp)
mkdir -p "$ROOT/logs"
SERVER_LOG="$ROOT/logs/server.log"
rm -f "$SERVER_LOG"

# Model alias: one name in Cursor for whatever model is running (config/server.env: CURSOR_MODEL_ALIAS, default "local")
MODEL_ALIAS="${CURSOR_MODEL_ALIAS:-local}"
# Build argv (host, port, n_gpu_layers, jinja from config)
argv=(-m "$MODEL_PATH" --alias "$MODEL_ALIAS" --host "${HOST:-127.0.0.1}" --port "${PORT}" --n-gpu-layers "${N_GPU_LAYERS:--1}" -c "${CONTEXT_SIZE:-262144}")
[[ -n "$MMPROJ_PATH" ]] && argv+=(--mmproj "$MMPROJ_PATH")
[[ -n "$THREADS" ]] && argv+=(--threads "$THREADS")
[[ "${JINJA:-1}" =~ ^(1|true|on|yes)$ ]] && argv+=(--jinja)
[[ -n "$TEMP" ]]    && argv+=(--temp "$TEMP")
[[ -n "$TOP_P" ]]   && argv+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]]   && argv+=(--top-k "$TOP_K")
[[ -n "$MIN_P" ]]   && argv+=(--min-p "$MIN_P")
[[ -n "$REPEAT_PENALTY" ]] && argv+=(--repeat-penalty "$REPEAT_PENALTY")
[[ -n "$SEED" ]]    && argv+=(--seed "$SEED")
[[ -n "$BATCH_SIZE" ]]  && argv+=(--batch-size "$BATCH_SIZE")
[[ -n "$UBATCH_SIZE" ]] && argv+=(--ubatch-size "$UBATCH_SIZE")
[[ -n "$FLASH_ATTN" ]]  && argv+=(--flash-attn "$FLASH_ATTN")
[[ -n "$CACHE_TYPE_K" ]] && argv+=(--cache-type-k "$CACHE_TYPE_K")
[[ -n "$CACHE_TYPE_V" ]] && argv+=(--cache-type-v "$CACHE_TYPE_V")
[[ -n "$VERBOSE" ]]     && argv+=(--verbose)
argv+=(--log-file "$SERVER_LOG")
# Optional chat template override (e.g. Qwen3 Coder tool-calling fix)
if [[ -n "$CHAT_TEMPLATE_FILE" ]]; then
  if [[ "$CHAT_TEMPLATE_FILE" != /* ]]; then
    CHAT_TEMPLATE_FILE="$ROOT/$CHAT_TEMPLATE_FILE"
  fi
  if [[ -f "$CHAT_TEMPLATE_FILE" ]]; then
    argv+=(--chat-template-file "$CHAT_TEMPLATE_FILE")
    echo "chat_template_file=$(basename "$CHAT_TEMPLATE_FILE")"
  fi
fi

echo "port=${PORT} ctx=${CONTEXT_SIZE:-262144} model=$(basename "$MODEL_PATH")${MMPROJ_PATH:+ mmproj=$(basename "$MMPROJ_PATH")}${VERBOSE:+ verbose=1}"
echo "log=$SERVER_LOG"
echo "API: http://${HOST:-127.0.0.1}:${PORT}/v1  (use in Cursor: $MODEL_ALIAS)"
exec "$LLAMA_SERVER" "${argv[@]}"
