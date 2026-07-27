#!/usr/bin/env bash
# Start vLLM OpenAI-compatible server. Consumes env from run_server.sh (after load_model_config).
# Requires: conda env with vllm (e.g. conda run -n vLLM vllm serve ...).
# Usage: called by run_server.sh when BACKEND=vllm; do not run standalone without env set.
#
# Ctrl+C (SIGINT): cleanup trap kills only this script's children (vllm | tee and their
# descendants, e.g. EngineCore). Never kills the shell, parent, or session.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Safe to kill? Only our own descendants; never self, parent, or init.
_safe_pid() {
  local pid=$1
  [[ -z "$pid" || ! "$pid" =~ ^[0-9]+$ ]] && return 1
  [[ "$pid" -le 1 || "$pid" -eq $$ || "$pid" -eq "${PPID:-0}" ]] && return 1
  return 0
}

# Kill process and all its descendants (e.g. vllm -> EngineCore workers). Never kill self/parent/init.
kill_tree_terminate() {
  local pid=$1
  _safe_pid "$pid" || return
  for c in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree_terminate "$c"
  done
  kill -TERM "$pid" 2>/dev/null || true
}
kill_tree_kill() {
  local pid=$1
  _safe_pid "$pid" || return
  for c in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree_kill "$c"
  done
  kill -KILL "$pid" 2>/dev/null || true
}

# On Ctrl+C / SIGTERM: kill only our vLLM pipeline and its descendants, then exit. Never touch session/parent.
cleanup() {
  trap - INT TERM
  echo "" >&2
  echo "Stopping vLLM and workers..." >&2
  # 1) Kill our background job's process group (vllm + tee). jobs -p gives one PID; use its PGID so the whole pipeline gets TERM.
  local jp
  for jp in $(jobs -p 2>/dev/null); do
    [[ -z "$jp" || ! "$jp" =~ ^[0-9]+$ ]] && continue
    if _safe_pid "$jp"; then
      local pgid
      pgid=$(ps -o pgid= -p "$jp" 2>/dev/null | tr -d ' ')
      if [[ -n "$pgid" && "$pgid" =~ ^[0-9]+$ && "$pgid" -gt 1 && "$pgid" -ne $$ ]]; then
        kill -TERM -"$pgid" 2>/dev/null || true
      else
        kill -TERM "$jp" 2>/dev/null || true
      fi
    fi
  done
  # 2) Recursively TERM only our direct children and their trees (vllm, tee, EngineCore, etc.). pgrep -P $$ = children of this script only.
  local pid
  for pid in $(pgrep -P $$ 2>/dev/null); do
    kill_tree_terminate "$pid"
  done
  sleep 2
  # 3) Recursively KILL any of our descendants still alive
  for pid in $(pgrep -P $$ 2>/dev/null); do
    kill_tree_kill "$pid"
  done
  wait 2>/dev/null || true
  exit 130
}
trap cleanup INT TERM

# Required
if [[ -z "${VLLM_MODEL:-}" ]]; then
  echo "VLLM_MODEL is not set. Use a model YAML with backend: vllm and vllm_model." >&2
  exit 1
fi
# Resolve relative paths (e.g. models/...) to absolute. Leave Hugging Face model IDs (e.g. org/name) as-is.
if [[ "$VLLM_MODEL" == models/* ]] || [[ "$VLLM_MODEL" == ./* ]] || [[ "$VLLM_MODEL" == ../* ]]; then
  VLLM_MODEL="$ROOT/$VLLM_MODEL"
fi

# MXFP4 models (e.g. GadflyII/GLM-4.7-Flash-MXFP4) require Marlin backend
if [[ "$VLLM_MODEL" == *MXFP4* ]]; then
  export VLLM_MXFP4_USE_MARLIN=1
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

# Run vllm from env directly (no conda run) so stdout/stderr stream to tee; conda run buffers when piped
VLLM_ENV=$(conda run -n "$CONDA_ENV" printenv CONDA_PREFIX 2>/dev/null) || true
[[ -z "$VLLM_ENV" ]] && VLLM_ENV="$HOME/.conda/envs/$CONDA_ENV"
if [[ ! -x "$VLLM_ENV/bin/vllm" ]]; then
  echo "vllm not found at $VLLM_ENV/bin/vllm" >&2
  exit 1
fi

argv=()
argv+=("$VLLM_ENV/bin/vllm" serve "$VLLM_MODEL")
argv+=(--host "$HOST" --port "$PORT" --served-model-name "$SERVED_NAME")

# vLLM serve CLI: https://docs.vllm.ai/en/latest/cli/serve.html
# No --temperature/--top-p; use --override-generation-config for default sampling (JSON).
[[ -n "${VLLM_TOOL_CALL_PARSER:-}" ]]    && argv+=(--tool-call-parser "$VLLM_TOOL_CALL_PARSER")
[[ -n "${VLLM_MAX_MODEL_LEN:-}" ]]       && argv+=(--max-model-len "$VLLM_MAX_MODEL_LEN")
[[ -n "${VLLM_TENSOR_PARALLEL:-}" ]]     && argv+=(--tensor-parallel-size "$VLLM_TENSOR_PARALLEL")
[[ -n "${VLLM_GPU_MEMORY_UTILIZATION:-}" ]] && argv+=(--gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION")
# Only pass --tokenizer for HF repo ids; for local paths vLLM loads tokenizer from model dir (avoids "Repo id must be in the form" error).
if [[ -n "${VLLM_TOKENIZER:-}" ]]; then
  if [[ "$VLLM_MODEL" == /* ]] || [[ "$VLLM_MODEL" == "$ROOT"/models/* ]]; then
    : # local path: do not pass --tokenizer
  else
    argv+=(--tokenizer "$VLLM_TOKENIZER")
  fi
fi
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
export PYTHONUNBUFFERED=1
export PATH="$VLLM_ENV/bin:$PATH"
# Run pipeline in background so the shell stays foreground and receives Ctrl+C (SIGINT).
# Otherwise the pipeline gets SIGINT and the trap never runs, so cleanup never executes.
"${argv[@]}" 2>&1 | tee "$SERVER_LOG" &
wait
