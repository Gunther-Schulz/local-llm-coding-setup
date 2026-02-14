#!/usr/bin/env bash
# Mode 3: Notebook LM via llama-server router mode. One process, embedding + chat on one port.
# Uses --models-dir; clients use model= bge-m3 (embeddings) and notebook-chat (chat).
# Usage: ./run_notebook.sh [--verbose] [--no-log-buffer]

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f "$ROOT/config/server.env" ]]; then
  echo "config/server.env not found" >&2
  exit 1
fi
set -a
source "$ROOT/config/server.env"
set +a

VERBOSE=""
NO_LOG_BUFFER=""
for arg in "$@"; do
  [[ "$arg" == "--verbose" ]] && VERBOSE=1
  [[ "$arg" == "--no-log-buffer" ]] && NO_LOG_BUFFER=1
done

PORT="${NOTEBOOK_CHAT_PORT:-8001}"
H="${HOST:-127.0.0.1}"

echo "Building notebook router dir (embedding + chat symlinks)..."
"$ROOT/scripts/build_notebook_router_dir.sh"

LLAMA_SERVER="${LLAMACPP_SERVER_BIN:-$ROOT/external/llama.cpp/build-cuda/bin/llama-server}"
if [[ "$LLAMA_SERVER" != /* ]]; then
  LLAMA_SERVER="$ROOT/$LLAMA_SERVER"
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found: $LLAMA_SERVER" >&2
  echo "Run: ./setup/install.sh  or  ./setup/build/llamacpp_cuda.sh" >&2
  exit 1
fi

# Logging (same as run_server.sh)
mkdir -p "$ROOT/logs"
SERVER_LOG="$ROOT/logs/server.log"
rm -f "$SERVER_LOG"
SERVER_LOG_TAIL_LINES="${SERVER_LOG_TAIL_LINES:-}"

ROUTER_DIR="$ROOT/models/.notebook-router"
echo "Starting notebook stack (router mode): one server on port $PORT"
echo "Embedding model: bge-m3  |  Chat model: notebook-chat"
echo "API: http://$H:$PORT/v1  (use model= bge-m3 for /v1/embeddings, notebook-chat for /v1/chat/completions)"
[[ -n "$VERBOSE" ]] && echo "verbose=1"
echo "log=$SERVER_LOG"
[[ -n "$SERVER_LOG_TAIL_LINES" ]] && echo "log_tail_lines=$SERVER_LOG_TAIL_LINES"

PRESET_INI="$ROOT/config/notebook-router-models.ini"
argv=(
  --models-dir "$ROUTER_DIR"
  --models-preset "$PRESET_INI"
  --models-max 2
  --host "$H"
  --port "$PORT"
  -c 131072
  --n-gpu-layers -1
  --threads 28
  --jinja
  --temp 1.0
  --top-p 0.95
  --top-k 40
  --min-p 0.01
  --seed 3407
  --batch-size 4096
  --ubatch-size 4096
)
[[ -n "$VERBOSE" ]] && argv+=(--verbose)
# When using tail wrapper we capture stdout/stderr only; do not add --log-file (would duplicate and grow unbounded)
# --no-log-buffer bypasses tail wrapper and writes directly to one log file
[[ -n "$NO_LOG_BUFFER" ]] && SERVER_LOG_TAIL_LINES=""
if [[ -z "$SERVER_LOG_TAIL_LINES" || ! "$SERVER_LOG_TAIL_LINES" =~ ^[0-9]+$ ]]; then
  argv+=(--log-file "$SERVER_LOG")
fi
if [[ -n "$SERVER_LOG_TAIL_LINES" && "$SERVER_LOG_TAIL_LINES" =~ ^[0-9]+$ ]]; then
  exec "$LLAMA_SERVER" "${argv[@]}" 2>&1 | "$ROOT/scripts/keep_last_n_log.sh" "$SERVER_LOG" "$SERVER_LOG_TAIL_LINES" 1
else
  exec "$LLAMA_SERVER" "${argv[@]}" >> "$SERVER_LOG" 2>&1
fi
