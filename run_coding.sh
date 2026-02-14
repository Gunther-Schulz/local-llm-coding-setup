#!/usr/bin/env bash
# Mode 2: Coding. One LLM; Cursor can talk to server (8001) or to proxy (8010) for tool_choice=required.
# With proxy: run ./run_coding.sh (server), then ./start-proxy.sh (proxy). Point Cursor at http://HOST:8010.
# Usage: ./run_coding.sh [--verbose] [--no-log-buffer]
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f "$ROOT/config/server.env" ]] || { echo "config/server.env not found" >&2; exit 1; }
set -a
source "$ROOT/config/server.env"
set +a
EXTRA=()
for arg in "$@"; do
  [[ "$arg" == "--verbose" ]] && EXTRA+=(--verbose)
  [[ "$arg" == "--no-log-buffer" ]] && EXTRA+=(--no-log-buffer)
done
exec "$ROOT/run_server.sh" "${EXTRA[@]}" "$CODING_MODEL" "$CODING_PORT"
