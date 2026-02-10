#!/usr/bin/env bash
# Build a long prompt by concatenating project files (proxy/ + stack/ + external/llama.cpp) for long-context benchmark.
# Output to stdout; optional MAX_CHARS to cap size (~4 chars per token rough).
# Usage: ./fill_context.sh [MAX_CHARS]
# Example: ./fill_context.sh 100000 > long_prompt.txt  # ~25k tokens
set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
MAX_CHARS="${1:-0}"

buf=""
total=0
while IFS= read -r -d '' f; do
  if [[ ! -f "$f" ]]; then continue; fi
  head="=== $f ===
"
  body="$(cat "$f")"
  block="$head$body

"
  if [[ -n "$MAX_CHARS" && "$MAX_CHARS" -gt 0 ]]; then
    need=$((MAX_CHARS - total))
    if [[ ${#block} -gt $need ]]; then
      block="${block:0:$need}"
      buf+="$block"
      break
    fi
  fi
  buf+="$block"
  total=${#buf}
done < <(find "$ROOT/proxy" "$ROOT/stack" "$ROOT/external/llama.cpp" -type f \( -name "*.py" -o -name "*.sh" -o -name "*.c" -o -name "*.cpp" -o -name "*.h" \) -print0 2>/dev/null | sort -z)

echo -n "$buf"
