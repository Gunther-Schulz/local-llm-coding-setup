#!/usr/bin/env bash
# Start the chat proxy (tool_choice=required when tools present). Run after the server.
# Usage: ./start-proxy.sh [--debug]
#   Or: PROXY_DEBUG=1 ./start-proxy.sh   or   ./start-proxy.sh --debug
# Point Cursor at http://HOST:PROXY_PORT (e.g. 8010) instead of the server port.
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f "$ROOT/config/server.env" ]] || { echo "config/server.env not found" >&2; exit 1; }
set -a
source "$ROOT/config/server.env"
set +a
BACKEND_URL="${BACKEND_URL:-http://${HOST:-127.0.0.1}:${CODING_PORT:-8001}}"
PROXY_PORT="${PROXY_PORT:-${CODING_PROXY_PORT:-8010}}"
export BACKEND_URL PROXY_PORT
[[ -n "$PROXY_DEBUG" ]] && export PROXY_DEBUG
# Loop limit: stop after same tool call repeated N times (config in config/server.env)
PROXY_MAX_IDENTICAL_TOOL_CALLS="${PROXY_MAX_IDENTICAL_TOOL_CALLS:-${CODING_PROXY_MAX_IDENTICAL_TOOL_CALLS}}"
[[ -n "$PROXY_MAX_IDENTICAL_TOOL_CALLS" ]] && export PROXY_MAX_IDENTICAL_TOOL_CALLS
# Similar limit: stop after same file Read or same Grep pattern N rounds
PROXY_MAX_SIMILAR_TOOL_CALLS="${PROXY_MAX_SIMILAR_TOOL_CALLS:-${CODING_PROXY_MAX_SIMILAR_TOOL_CALLS}}"
[[ -n "$PROXY_MAX_SIMILAR_TOOL_CALLS" ]] && export PROXY_MAX_SIMILAR_TOOL_CALLS
exec python3 "$ROOT/scripts/chat_proxy.py" "$@"
