#!/usr/bin/env bash
# Mode 1: Pure chat. One LLM; client talks to server (or proxy when PURE_CHAT_PROXY_PORT is set).
# Usage: ./run_chat.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f "$ROOT/config/server.env" ]] || { echo "config/server.env not found" >&2; exit 1; }
set -a
source "$ROOT/config/server.env"
set +a
exec "$ROOT/run_server.sh" "$PURE_CHAT_MODEL" "$PURE_CHAT_PORT"
