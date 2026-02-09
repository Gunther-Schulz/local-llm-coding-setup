#!/usr/bin/env bash
# Start llama-server for one Qwen3-Coder-Next benchmark scenario.
# Usage: run_server.sh SCENARIO [PORT]
# Example: run_server.sh q2_full 18999
set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
SCENARIO="${1:?Usage: run_server.sh SCENARIO [PORT]}"
PORT="${2:-18999}"

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
# Columns: scenario_name|model_path|moe_ot|cache_k|n_gpu_layers (optional; empty = use env; integer or 25%/50%/75%)
CFG="$BENCH_DIR/scenarios.cfg"
model_path=""
moe_ot=""
cache_k=""
n_gpu_layers_cfg=""
while IFS= read -r line; do
  [[ "$line" =~ ^# ]] && continue
  [[ -z "$line" ]] && continue
  name="${line%%|*}"; rest="${line#*|}"
  if [[ "$name" == "$SCENARIO" ]]; then
    IFS='|' read -r model_path moe_ot cache_k n_gpu_layers_cfg <<< "$rest"
    moe_ot="${moe_ot//[$'\r\n']}"; cache_k="${cache_k//[$'\r\n']}"; n_gpu_layers_cfg="${n_gpu_layers_cfg//[$'\r\n']}"
    break
  fi
done < "$CFG"

if [[ -z "$model_path" ]]; then
  echo "Unknown scenario: $SCENARIO"
  echo "Known: $(grep -v '^#' "$CFG" | grep -v '^$' | cut -d'|' -f1 | tr '\n' ' ')"
  exit 1
fi

full_path="$ROOT/$model_path"
if [[ ! -f "$full_path" ]]; then
  # Allow single .gguf in scenario dir (download may name file differently)
  dir="${full_path%/*}"
  if [[ -d "$dir" ]]; then
    one=$(find "$dir" -maxdepth 1 -name "*.gguf" -print -quit)
    if [[ -n "$one" ]]; then
      full_path="$one"
    fi
  fi
fi
if [[ ! -f "$full_path" ]]; then
  echo "Model not found: $full_path"
  echo "Run ./benchmark/next/download.sh first."
  exit 1
fi

# N_GPU_LAYERS: -1 = all on GPU; 0 = all on CPU. Scenario can override with 5th column (integer or 25%/50%/75%).
# When env N_GPU_LAYERS=0 (CPU pass), we keep 0. Otherwise use scenario n_gpu_layers if set.
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
else
  N_GPU_LAYERS="${N_GPU_LAYERS:--1}"
fi
argv=("$LLAMA_SERVER" -m "$full_path" --host "127.0.0.1" --port "$PORT" --n-gpu-layers "$N_GPU_LAYERS" --jinja)
# Context: fixed for benchmark (avoid --fit so we compare same -c)
argv+=(-c "32768")
[[ -n "$moe_ot" ]] && argv+=(-ot "$moe_ot")
[[ -n "$cache_k" ]] && argv+=(--cache-type-k "$cache_k")

echo "Scenario: $SCENARIO | port $PORT | n_gpu_layers=$N_GPU_LAYERS | model $full_path"
exec "${argv[@]}"
