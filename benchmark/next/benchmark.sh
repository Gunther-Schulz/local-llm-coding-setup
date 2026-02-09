#!/usr/bin/env bash
# Run all Qwen3-Coder-Next scenarios: start server -> measure tok/s -> stop.
# Default: short only (quick). Use --long to also run long-context tests.
set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
PORT=18999
# Default: short only. Pass --long to run both short and long context.
# Default: run GPU pass then CPU (system-only) pass for comparison. Set RUN_CPU=0 to skip CPU pass.
SHORT_ONLY="${SHORT_ONLY:-1}"
RUN_CPU="${RUN_CPU:-1}"
[[ "$1" == "--long" ]] && SHORT_ONLY=0
[[ "$1" == "--short-only" ]] && SHORT_ONLY=1

cd "$BENCH_DIR"
RESULTS_FILE="${BENCH_DIR}/results.txt"
if [[ "$SHORT_ONLY" == "1" ]]; then
  echo "Mode: short only (use --long to also run long-context tests)"
else
  echo "Mode: short + long context (--long)"
fi
[[ "$RUN_CPU" == "1" ]] && echo "Will run GPU then CPU (system-only) pass for comparison"
echo ""
scenarios=()
while IFS= read -r line; do
  [[ "$line" =~ ^# ]] && continue
  [[ -z "$line" ]] && continue
  name="${line%%|*}"
  scenarios+=("$name")
done < scenarios.cfg
results=()
results_cpu=()

# Write one result line to the results file (append). Call after each scenario so partial results survive crashes.
write_result_line() {
  local name="$1" short="$2" long="$3" ctx_short="$4" ctx_long="$5"
  printf "%-18s %12s %12s %12s %12s\n" "$name" "$short" "$long" "$ctx_short" "$ctx_long" >> "$RESULTS_FILE"
}

run_pass() {
  local backend_label="$1"
  local -n res_arr=$2
  res_arr=()
  for scenario in "${scenarios[@]}"; do
    echo "--- $scenario ($backend_label) ---"
    "$BENCH_DIR/run_server.sh" "$scenario" "$PORT" &
    pid=$!
    trap "kill $pid 2>/dev/null; exit 1" INT TERM
    for i in {1..60}; do
      if curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then break; fi
      sleep 1
    done
    if ! curl -s "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
      echo "Server did not become ready for $scenario"
      kill $pid 2>/dev/null
      res_arr+=("$scenario|FAIL|-|-|-")
      write_result_line "$scenario" "FAIL" "-" "-" "-"
      continue
    fi
    echo "  Waiting for model to load..."
    model_ready=""
    for _ in {1..60}; do
      code=$(curl -s -o /dev/null -w "%{http_code}" -X POST "http://127.0.0.1:$PORT/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d '{"model":"qwen3-coder-next","messages":[{"role":"user","content":"x"}],"max_tokens":1}' 2>/dev/null || echo "000")
      if [[ "$code" == "200" ]]; then
        model_ready=1
        break
      fi
      sleep 5
    done
    if [[ -z "$model_ready" ]]; then
      echo "  Model did not become ready (got HTTP $code); skipping measure"
      kill $pid 2>/dev/null
      wait $pid 2>/dev/null || true
      res_arr+=("$scenario|FAIL|-|-|-")
      write_result_line "$scenario" "FAIL" "-" "-" "-"
      continue
    fi
    out=$(python3 "$BENCH_DIR/measure.py" --port "$PORT" 2>&1) || true
    tok_s_short=""
    short_ctx=""
    if [[ "$out" =~ tok/s=([0-9.]+) ]]; then
      tok_s_short="${BASH_REMATCH[1]}"
      [[ "$out" =~ prompt_tokens=([0-9]+) ]] && short_ctx="${BASH_REMATCH[1]}"
    else
      [[ -n "$out" ]] && echo "  measure: $out"
    fi
    tok_s_long=""
    long_ctx=""
    if [[ "$SHORT_ONLY" != "1" ]]; then
      fill="$BENCH_DIR/.long_prompt.txt"
      "$BENCH_DIR/fill_context.sh" 100000 > "$fill" 2>/dev/null || true
      if [[ -s "$fill" ]]; then
        out_long=$(python3 "$BENCH_DIR/measure.py" --port "$PORT" --prompt-file "$fill" 2>&1) || true
        if [[ "$out_long" =~ tok/s=([0-9.]+) ]]; then
          tok_s_long="${BASH_REMATCH[1]}"
          [[ "$out_long" =~ prompt_tokens=([0-9]+) ]] && long_ctx="${BASH_REMATCH[1]}"
        else
          [[ -n "$out_long" ]] && echo "  measure (long): $out_long"
        fi
      fi
    fi
    kill $pid 2>/dev/null
    wait $pid 2>/dev/null || true
    res_arr+=("$scenario|${tok_s_short:--}|${tok_s_long:--}|${short_ctx:--}|${long_ctx:--}")
    write_result_line "$scenario" "${tok_s_short:--}" "${tok_s_long:--}" "${short_ctx:--}" "${long_ctx:--}"
    echo "  short: ${tok_s_short:--} tok/s ctx=${short_ctx:--}  long: ${tok_s_long:--} tok/s ctx=${long_ctx:--}"
  done
}

# Start results file (overwrite) so partial results survive crashes
if [[ "$SHORT_ONLY" == "1" ]]; then
  MODE_STR="short only"
else
  MODE_STR="short + long (--long)"
fi
{
  echo "Qwen3-Coder-Next benchmark — $(date -Iseconds 2>/dev/null || date '+%Y-%m-%d %H:%M:%S')"
  echo "Mode: $MODE_STR"
  echo ""
  echo "=== GPU ==="
  printf "%-18s %12s %12s %12s %12s\n" "Scenario" "Short tok/s" "Long tok/s" "Short ctx" "Long ctx"
  printf "%-18s %12s %12s %12s %12s\n" "-------" "----------" "---------" "---------" "--------"
} > "$RESULTS_FILE"

# GPU pass (default N_GPU_LAYERS=-1)
run_pass "GPU" results

# Optional CPU (system-only) pass
if [[ "$RUN_CPU" == "1" ]]; then
  echo ""
  echo "=== CPU (system only) pass ==="
  {
    echo ""
    echo "=== CPU (system only) ==="
    printf "%-18s %12s %12s %12s %12s\n" "Scenario" "Short tok/s" "Long tok/s" "Short ctx" "Long ctx"
    printf "%-18s %12s %12s %12s %12s\n" "-------" "----------" "---------" "---------" "--------"
  } >> "$RESULTS_FILE"
  export N_GPU_LAYERS=0
  run_pass "CPU" results_cpu
  unset N_GPU_LAYERS
fi

echo ""
echo "=== Summary ==="
cat "$RESULTS_FILE"
echo ""
echo "Results written to $RESULTS_FILE"
