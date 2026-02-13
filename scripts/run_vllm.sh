#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server. Consumes env from run_server.sh (after load_model_config).
# Requires: conda env with vllm (e.g. conda run -n vLLM vllm serve ...).
# Usage: called by run_server.sh when BACKEND=vllm; do not run standalone without env set.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Required
if [[ -z "${VLLM_MODEL:-}" ]]; then
  echo "VLLM_MODEL is not set. Use a model YAML with backend: vllm and vllm_model." >&2
  exit 1
fi
# Resolve relative paths (e.g. models/...) to absolute so vllm sees the file regardless of CWD
if [[ "$VLLM_MODEL" != /* ]] && [[ "$VLLM_MODEL" != *:* ]]; then
  VLLM_MODEL="$ROOT/$VLLM_MODEL"
fi

CONDA_ENV="${VLLM_CONDA_ENV:-vLLM}"
if ! command -v conda &>/dev/null; then
  echo "conda not found. Install Miniconda/Anaconda or activate conda." >&2
  exit 1
fi
if ! conda run -n "$CONDA_ENV" python -c "import vllm" 2>/dev/null; then
  echo "vllm not found in conda env '$CONDA_ENV'. Run: conda activate $CONDA_ENV && pip install vllm" >&2
  exit 1
fi

PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
# Name shown in /v1/models and responses (Cursor uses this)
SERVED_NAME="${CURSOR_MODEL_ALIAS:-${ACTIVE_MODEL:-local}}"

# Log: same as llama path (run_server.sh sets SERVER_LOG)
mkdir -p "$ROOT/logs"
SERVER_LOG="${SERVER_LOG:-$ROOT/logs/server.log}"
rm -f "$SERVER_LOG"

argv=()
argv+=(conda run -n "$CONDA_ENV" vllm serve "$VLLM_MODEL")
argv+=(--host "$HOST" --port "$PORT" --served-model-name "$SERVED_NAME")

# vLLM serve CLI: https://docs.vllm.ai/en/latest/cli/serve.html
# No --temperature/--top-p; use --override-generation-config for default sampling (JSON).
[[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]    && argv+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
[[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]       && argv+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
[[ -n "${VLLM_TENSOR_PARALLEL:-}" ]]     && argv+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL")
[[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]] && argv+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
[[ -n "${VLLM_TOKENIZER:-}" ]]           && argv+=(--tokenizer "$VLLM_TOKENIZER")
if [[ -n "${TEMP:-}" ]] || [[ -n "${TOP_P:-}" ]]; then
  gen_parts=()
  [[ -n "${TEMP:-}" ]]  && gen_parts+=("\"temperature\": ${TEMP}")
  [[ -n "${TOP_P:-}" ]] && gen_parts+=("\"top_p\": ${TOP_P}")
  argv+=(--override-generation-config "{$(IFS=,; echo "${gen_parts[*]}")}")
fi

# Extra args from YAML (vllm_serve_extra), space-separated
if [[ -n "${VLLM_SERVE_EXTRA:-}" ]]; then
  read -ra extra <<< "$VLLM_SERVE_EXTRA"
  argv+=("${extra[@]}")
fi

echo "port=${PORT} backend=vllm model=${VLLM_MODEL} served_as=${SERVED_NAME}"
echo "log=$SERVER_LOG"
echo "API: http://${HOST}:${PORT}/v1  (use in Cursor: $SERVED_NAME)"
exec "${argv[@]}" >> "$SERVER_LOG" 2>&1
