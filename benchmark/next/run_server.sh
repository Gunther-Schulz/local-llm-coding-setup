#!/usr/bin/env bash
# Start llama-server for one benchmark scenario. Config from config/models/<model_key>.yaml.
# Usage: run_server.sh SCENARIO [PORT]
# Example: run_server.sh qwen3-coder-next-mxfp4 18999
set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
SCENARIO="${1:?Usage: run_server.sh SCENARIO [PORT]}"
PORT="${2:-18999}"

# Optional: same as main run_server.sh (LLAMA_THREADS overrides per-model YAML threads)
[[ -f "$ROOT/config/server.env" ]] && set -a && source "$ROOT/config/server.env" && set +a

LLAMA_SERVER="${LLAMACPP_SERVER_BIN:-$ROOT/external/llama.cpp/build-cuda/bin/llama-server}"
if [[ ! -x "$LLAMA_SERVER" && "$LLAMA_SERVER" != /* ]]; then
  LLAMA_SERVER="$ROOT/$LLAMA_SERVER"
fi

if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found: $LLAMA_SERVER"
  echo "Set LLAMACPP_SERVER_BIN or run ./setup/build/llamacpp_cuda.sh"
  exit 1
fi

# Parse scenario from scenarios.cfg
# Columns: scenario_name|model_key|moe_ot|cache_k|n_gpu_layers[|api_model][|ctx_override]
# Empty n_gpu_layers = use YAML (-1); llama-server defaults to --fit on, so it auto-reduces to fit VRAM. Override for fixed split.
# ctx_override: 0 = use model default/full context (-c 0 in llama-server; fit won't reduce). Empty = use YAML/BENCHMARK_CTX.
# model_key = config/models/<model_key>.yaml (same as ACTIVE_MODEL in main stack)
CFG="$BENCH_DIR/scenarios.cfg"
model_key=""
moe_ot=""
cache_k=""
n_gpu_layers_cfg=""
ctx_override=""
while IFS= read -r line; do
  [[ "$line" =~ ^# ]] && continue
  [[ -z "$line" ]] && continue
  name="${line%%|*}"; rest="${line#*|}"
  if [[ "$name" == "$SCENARIO" ]]; then
    IFS='|' read -r model_key moe_ot cache_k n_gpu_layers_cfg _ ctx_override <<< "$rest"
    model_key="${model_key//[$'\r\n']}"; moe_ot="${moe_ot//[$'\r\n']}"; cache_k="${cache_k//[$'\r\n']}"; n_gpu_layers_cfg="${n_gpu_layers_cfg//[$'\r\n']}"; ctx_override="${ctx_override//[$'\r\n']}"
    break
  fi
done < "$CFG"

if [[ -z "$model_key" ]]; then
  echo "Unknown scenario: $SCENARIO"
  echo "Known: $(grep -v '^#' "$CFG" | grep -v '^$' | cut -d'|' -f1 | tr '\n' ' ')"
  exit 1
fi

# Load config from config/models/<model_key>.yaml (same as run_server.sh)
set -a
eval "$("$ROOT/scripts/load_model_config.sh" "$model_key")"
set +a
# config/server.env override for CPU threads (overrides YAML when set)
[[ -n "${LLAMA_THREADS:-}" ]] && THREADS="$LLAMA_THREADS"

MODEL_PATH="$ROOT/models/${model_key}/${GGUF}"
if [[ ! -f "$MODEL_PATH" ]]; then
  dir="$ROOT/models/${model_key}"
  if [[ -d "$dir" ]]; then
    one=$(find "$dir" -maxdepth 2 -name "*.gguf" -print -quit)
    if [[ -n "$one" ]]; then
      MODEL_PATH="$one"
    fi
  fi
fi
if [[ ! -f "$MODEL_PATH" ]]; then
  echo "Model not found: $MODEL_PATH"
  echo "Ensure models are in models/${model_key}/ (see config/models/${model_key}.yaml gguf:)"
  exit 1
fi

# N_GPU_LAYERS: from YAML unless scenario overrides (5th column) or env N_GPU_LAYERS=0 for CPU pass
BENCHMARK_N_LAYERS="${BENCHMARK_N_LAYERS:-80}"
if [[ "$N_GPU_LAYERS" == "0" ]]; then
  N_GPU_LAYERS=0
elif [[ -n "$n_gpu_layers_cfg" ]]; then
  if [[ "$n_gpu_layers_cfg" == *% ]]; then
    pct="${n_gpu_layers_cfg%\%}"
    N_GPU_LAYERS=$(( (BENCHMARK_N_LAYERS * pct) / 100 ))
  else
    N_GPU_LAYERS="$n_gpu_layers_cfg"
  fi
fi
# CONTEXT_SIZE from YAML; BENCHMARK_CTX overrides. Scenario ctx_override=0 => -c 0 (use model default/full context; fit won't reduce).
if [[ "$ctx_override" == "0" ]]; then
  ctx=0
else
  ctx="${BENCHMARK_CTX:-${CONTEXT_SIZE:-32768}}"
fi

argv=("$LLAMA_SERVER" -m "$MODEL_PATH" --host "127.0.0.1" --port "$PORT" --n-gpu-layers "${N_GPU_LAYERS:--1}" -c "$ctx")
[[ -n "${BENCHMARK_THREADS:-$THREADS}" ]] && argv+=(--threads "${BENCHMARK_THREADS:-$THREADS}")
[[ "${JINJA:-1}" =~ ^(1|true|on|yes)$ ]] && argv+=(--jinja)
[[ -n "$TEMP" ]]    && argv+=(--temp "$TEMP")
[[ -n "$TOP_P" ]]   && argv+=(--top-p "$TOP_P")
[[ -n "$TOP_K" ]]   && argv+=(--top-k "$TOP_K")
[[ -n "$MIN_P" ]]   && argv+=(--min-p "$MIN_P")
[[ -n "$SEED" ]]    && argv+=(--seed "$SEED")
[[ -n "$BATCH_SIZE" ]]  && argv+=(--batch-size "$BATCH_SIZE")
[[ -n "$UBATCH_SIZE" ]] && argv+=(--ubatch-size "$UBATCH_SIZE")
[[ -n "$moe_ot" ]]  && argv+=(-ot "$moe_ot")
[[ -n "$cache_k" ]] && argv+=(--cache-type-k "$cache_k")

echo "Scenario: $SCENARIO | port $PORT | ctx=$ctx | n_gpu_layers=${N_GPU_LAYERS:--1} | threads=${BENCHMARK_THREADS:-${THREADS:-default}} | model $MODEL_PATH"
exec "${argv[@]}"
