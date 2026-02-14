#!/usr/bin/env bash
# Start the LLM proxy. Loads config/server.env; single backend or Code+Vision routing.
# Single mode: BACKEND_URL (default http://HOST:8001), PROXY_PORT (8010).
# Code+Vision: CODE_VISION_VISION_PORT (8002), CODE_VISION_CODING_PORT (8001); image → vision, else coding.
# Usage: ./start-proxy.sh [--debug]
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
[[ -f "$ROOT/config/server.env" ]] || { echo "config/server.env not found" >&2; exit 1; }
set -a
source "$ROOT/config/server.env"
set +a
export HOST="${HOST:-127.0.0.1}"
export BACKEND_URL="${BACKEND_URL:-http://${HOST}:8001}"
export PROXY_PORT="${PROXY_PORT:-8010}"
exec python3 -m proxy "$@"
