#!/usr/bin/env bash
# Wait until a port is accepting connections. Usage: wait_for_port.sh [HOST] PORT [MAX_SEC]
# Exits 0 when port is reachable (curl to http://HOST:PORT/v1/models), 1 on timeout.
set -e
HOST="${1:-127.0.0.1}"
PORT="${2:-$1}"
MAX_SEC="${3:-120}"
if [[ -z "$PORT" ]] || [[ "$PORT" == "$HOST" ]]; then
  echo "Usage: wait_for_port.sh [HOST] PORT [MAX_SEC]" >&2
  exit 1
fi
elapsed=0
until curl -sS -o /dev/null --connect-timeout 2 "http://${HOST}:${PORT}/v1/models" 2>/dev/null; do
  if [[ $elapsed -ge $MAX_SEC ]]; then
    echo "Timeout waiting for port $PORT" >&2
    exit 1
  fi
  sleep 1
  ((elapsed += 1)) || true
done
echo "Port $PORT ready (${elapsed}s)" >&2
