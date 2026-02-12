#!/usr/bin/env bash
# Mode 3: Notebook LM via llama-server router mode. One process, embedding + chat on one port.
# Uses --models-dir; clients use model= bge-m3 (embeddings) and notebook-chat (chat).
# Usage: ./run_notebook.sh

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

ROUTER_DIR="$ROOT/models/.notebook-router"
echo "Starting notebook stack (router mode): one server on port $PORT"
echo "Embedding model: bge-m3  |  Chat model: notebook-chat"
echo "API: http://$H:$PORT/v1  (use model= bge-m3 for /v1/embeddings, notebook-chat for /v1/chat/completions)"
PRESET_INI="$ROOT/config/notebook-router-models.ini"
exec "$LLAMA_SERVER" \
  --models-dir "$ROUTER_DIR" \
  --models-preset "$PRESET_INI" \
  --models-max 2 \
  --host "$H" \
  --port "$PORT" \
  -c 131072 \
  --n-gpu-layers -1 \
  --threads 28 \
  --jinja \
  --temp 1.0 \
  --top-p 0.95 \
  --top-k 40 \
  --min-p 0.01 \
  --seed 3407 \
  --batch-size 4096 \
  --ubatch-size 4096
