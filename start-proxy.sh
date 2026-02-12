#!/bin/bash
# Start the tool proxy. Backend = chat server (always 8001 when one mode active). Used by run_coding.sh.
# With no args: uses config/server.env (HOST), BACKEND_URL and PROXY_PORT from env or defaults.
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
[[ -f "$ROOT/config/server.env" ]] && set -a && source "$ROOT/config/server.env" && set +a
BACKEND_URL="${BACKEND_URL:-http://${HOST:-127.0.0.1}:8001}"
PROXY_PORT="${PROXY_PORT:-8010}"
echo "Starting tool proxy server..."
source /opt/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
conda activate llm 2>/dev/null || true
cd "$ROOT/tool-proxy"
VERBOSE=
[[ -n "$DEBUG" ]] && VERBOSE="--verbose"
if [ $# -eq 0 ]; then
    echo "  Backend: $BACKEND_URL  Port: $PROXY_PORT"
    python server.py --port "$PROXY_PORT" --backend-url "$BACKEND_URL" --config config/default_rules.yaml $VERBOSE
else
    python server.py "$@"
fi