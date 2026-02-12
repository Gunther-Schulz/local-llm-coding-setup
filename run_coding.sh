#!/usr/bin/env bash
# Mode 2: Coding. One LLM; Cursor talks to server (or to proxy when ready — run start-proxy.sh separately).
# Proxy not started automatically yet; when ready, run: BACKEND_URL=http://HOST:8001 PROXY_PORT=8010 ./start-proxy.sh
# Usage: ./run_coding.sh
set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"
[[ -f "$ROOT/config/server.env" ]] || { echo "config/server.env not found" >&2; exit 1; }
set -a
source "$ROOT/config/server.env"
set +a
exec "$ROOT/run_server.sh" "$CODING_MODEL" "$CODING_PORT"
