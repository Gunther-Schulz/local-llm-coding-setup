#!/usr/bin/env bash
# Mode 3: Notebook LM. Embedding server first (8002), then chat server (8001). Proxy always points at 8001.
# Usage: ./run_notebook.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f "$ROOT/config/server.env" ]] || { echo "config/server.env not found" >&2; exit 1; }
set -a
source "$ROOT/config/server.env"
set +a

H="${HOST:-127.0.0.1}"
echo "Starting notebook stack: embedding $EMBEDDING_MODEL on $EMBEDDING_PORT, then chat $NOTEBOOK_CHAT_MODEL on $NOTEBOOK_CHAT_PORT"
"$ROOT/run_server.sh" "$EMBEDDING_MODEL" "$EMBEDDING_PORT" &
"$ROOT/scripts/wait_for_port.sh" "$H" "$EMBEDDING_PORT" 60
echo "Embedding ready. Starting chat server on $NOTEBOOK_CHAT_PORT (proxy backend)."
"$ROOT/run_server.sh" "$NOTEBOOK_CHAT_MODEL" "$NOTEBOOK_CHAT_PORT" &
"$ROOT/scripts/wait_for_port.sh" "$H" "$NOTEBOOK_CHAT_PORT" 120
echo "Notebook stack up. Embedding: http://$H:$EMBEDDING_PORT/v1  Chat: http://$H:$NOTEBOOK_CHAT_PORT/v1"
echo "Open Notebook: embedding API = http://$H:$EMBEDDING_PORT  chat API = http://$H:$NOTEBOOK_CHAT_PORT"
