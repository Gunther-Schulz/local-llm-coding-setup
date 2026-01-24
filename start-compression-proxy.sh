#!/bin/bash
ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

_conda_sh=""
for d in "$ROOT/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.miniconda3" "/workspace/miniconda3"; do
  if [ -f "${d}/etc/profile.d/conda.sh" ]; then
    _conda_sh="${d}/etc/profile.d/conda.sh"
    break
  fi
done
[ -z "$_conda_sh" ] && command -v conda &>/dev/null && {
  _base=$(conda info --base 2>/dev/null)
  [ -n "$_base" ] && [ -f "${_base}/etc/profile.d/conda.sh" ] && _conda_sh="${_base}/etc/profile.d/conda.sh"
}
if [ -n "$_conda_sh" ]; then
  # shellcheck disable=SC1090
  . "$_conda_sh"
  conda activate llm
elif [ -n "$VIRTUAL_ENV" ]; then
  :
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/venv/bin/activate"
else
  echo "ERROR: No conda (env llm) or venv. Create one and run ./install-deps.sh first."
  exit 1
fi

LOG_FILE="$ROOT/compression-proxy.log"

# Clear log file on start
> "$LOG_FILE"

# Check for debug flag
DEBUG_FLAG=""
if [ "$1" = "-d" ] || [ "$1" = "--debug" ]; then
    DEBUG_FLAG="--debug"
    echo "🚀 Starting compression proxy in DEBUG mode..."
else
    echo "🚀 Starting compression proxy..."
    echo "  (use: ./start-compression-proxy.sh --debug for full logging)"
fi

# Load model metadata from environment (if set by start-all-vllm.sh) or saved selection
CURRENT_MODEL_FILE="$ROOT/.current-model"

if [[ -n "$MODEL_TOOL_FORMAT" && -n "$MODEL_MAX_CONTEXT" ]]; then
    # Environment variables already set (by start-all-vllm.sh or manual export)
    echo "Using model config from environment:"
    echo "  Tool format: ${MODEL_TOOL_FORMAT}"
    echo "  Max context: ${MODEL_MAX_CONTEXT}"
elif [[ -f "$CURRENT_MODEL_FILE" ]]; then
    # No environment vars set, load from saved model selection
    SELECTED_MODEL=$(cat "$CURRENT_MODEL_FILE")
    
    # Source the model selector library to get config
    source "$ROOT/lib/model-selector.sh"
    
    # Load and export config
    if export_model_config "$SELECTED_MODEL" 2>/dev/null; then
        echo "Loaded configuration for model: $SELECTED_MODEL"
    else
        echo "⚠️  Warning: Could not load model config, using defaults"
        export MODEL_TOOL_FORMAT="auto"
        export MODEL_MAX_CONTEXT="32768"
    fi
else
    # No env vars and no saved selection, use defaults
    echo "⚠️  Warning: No model config found, using defaults"
    export MODEL_TOOL_FORMAT="auto"
    export MODEL_MAX_CONTEXT="32768"
fi

echo "  Backend: http://localhost:8000"
echo "  Model context: ${MODEL_MAX_CONTEXT} tokens"
echo "  Tool format: ${MODEL_TOOL_FORMAT}"
echo "  Log file: $LOG_FILE (cleared)"
python3 compression_proxy.py $DEBUG_FLAG >> "$LOG_FILE" 2>&1

