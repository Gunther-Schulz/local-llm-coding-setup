#!/usr/bin/env bash
# Start the minimal chat proxy (log + forward only).
# Backend: BACKEND_URL (default http://127.0.0.1:8001). Listen: PROXY_PORT (8010).
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
export BACKEND_URL="${BACKEND_URL:-http://127.0.0.1:8001}"
export PROXY_PORT="${PROXY_PORT:-8010}"
exec python3 scripts/chat_proxy.py
