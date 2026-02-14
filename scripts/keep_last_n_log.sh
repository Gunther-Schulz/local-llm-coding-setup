#!/usr/bin/env bash
# Reads stdin line-by-line, keeps only the last N lines in memory (via temp file),
# and writes them to the target log file periodically. Use to cap server.log size
# while keeping verbose logging for recent debugging.
#
# Usage: some_command 2>&1 | keep_last_n_log.sh LOG_FILE [MAX_LINES] [FLUSH_EVERY]
#   LOG_FILE    = path to output log (e.g. logs/server.log)
#   MAX_LINES   = max lines to keep (default: 50000)
#   FLUSH_EVERY = write to LOG_FILE every this many lines (default: 500)
#
# Env: SERVER_LOG_TAIL_LINES overrides MAX_LINES if set.
set -e
LOG_FILE="${1:?Usage: keep_last_n_log.sh LOG_FILE [MAX_LINES] [FLUSH_EVERY]}"
MAX_LINES="${2:-${SERVER_LOG_TAIL_LINES:-50000}}"
FLUSH_EVERY="${3:-500}"

[[ "$MAX_LINES" =~ ^[0-9]+$ ]] || MAX_LINES=50000
[[ "$FLUSH_EVERY" =~ ^[0-9]+$ ]] || FLUSH_EVERY=500

BUF_FILE="${LOG_FILE}.buf"
rm -f "$BUF_FILE"
touch "$BUF_FILE"
line_count=0

while IFS= read -r line || [[ -n "$line" ]]; do
  echo "$line" >> "$BUF_FILE"
  (( line_count++ )) || true
  if (( line_count >= MAX_LINES )); then
    tail -n "$MAX_LINES" "$BUF_FILE" > "${BUF_FILE}.tmp"
    mv "${BUF_FILE}.tmp" "$BUF_FILE"
    line_count=$MAX_LINES
  fi
  if (( line_count % FLUSH_EVERY == 0 )); then
    cp "$BUF_FILE" "$LOG_FILE"
  fi
done

# final flush
cp "$BUF_FILE" "$LOG_FILE"
rm -f "$BUF_FILE" "${BUF_FILE}.tmp"
