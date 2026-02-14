#!/usr/bin/env bash
# Mode 4: Code + Vision. Two models: vision (image → text) + coding (chat/tools). Like Mode 3 has embedding + chat.
# Starts vision server on CODE_VISION_VISION_PORT (8002), coding server on CODE_VISION_CODING_PORT (8001).
# Usage: ./run_code_vision.sh [--verbose]
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
done
VISION_PORT="${CODE_VISION_VISION_PORT:-8002}"
CODING_PORT="${CODE_VISION_CODING_PORT:-8001}"
CODE_MODEL="${CODE_VISION_CODING_MODEL:-$CODING_MODEL}"
echo "Mode 4: vision ${CODE_VISION_VISION_MODEL} on $VISION_PORT, coding ${CODE_MODEL} on $CODING_PORT"
echo "Starting vision server on $VISION_PORT (background)..."
"$ROOT/run_server.sh" "${EXTRA[@]}" "${CODE_VISION_VISION_MODEL:?CODE_VISION_VISION_MODEL not set}" "$VISION_PORT" &
VISION_PID=$!
trap "kill $VISION_PID 2>/dev/null || true" EXIT
sleep 2
echo "Starting coding server on $CODING_PORT (foreground)..."
exec "$ROOT/run_server.sh" "${EXTRA[@]}" "${CODE_MODEL:?CODE_VISION_CODING_MODEL or CODING_MODEL not set}" "$CODING_PORT"
