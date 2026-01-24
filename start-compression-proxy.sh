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
    export DEBUG=1
    echo "🚀 Starting compression proxy in DEBUG mode..."
else
    echo "🚀 Starting compression proxy..."
    echo "  (use: ./start-compression-proxy.sh --debug for full logging)"
fi

# Load config manager and model selector
source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/model-selector.sh"

# Load model configuration if not already set by start-all-vllm.sh
if [[ -z "$MODEL_TOOL_FORMAT" || -z "$MODEL_MAX_CONTEXT" ]]; then
    # Get current model from centralized config
    CURRENT_MODEL=$(get_current_model)
    
    if [[ -n "$CURRENT_MODEL" ]]; then
        # Export model configuration
        if export_model_config "$CURRENT_MODEL" >/dev/null 2>&1; then
            echo "Loaded configuration for model: $CURRENT_MODEL"
            
            # Load context mode and update MODEL_MAX_CONTEXT if extended
            EXTENDED_CONTEXT_MODE=$(get_extended_context_mode)
            if [[ "$EXTENDED_CONTEXT_MODE" == "1" && -n "$MODEL_EXTENDED_CONTEXT" ]]; then
                export MODEL_MAX_CONTEXT="$MODEL_EXTENDED_CONTEXT"
                echo "  Context mode: Extended (${MODEL_MAX_CONTEXT} tokens)"
            else
                echo "  Context mode: Normal (${MODEL_MAX_CONTEXT} tokens)"
            fi
        else
            echo "⚠️  Warning: Could not load model config, using defaults"
            export MODEL_TOOL_FORMAT="auto"
            export MODEL_MAX_CONTEXT="32768"
        fi
    else
        echo "⚠️  Warning: No model selected, using defaults"
        echo "   Run: ./select-model.sh to configure"
        export MODEL_TOOL_FORMAT="auto"
        export MODEL_MAX_CONTEXT="32768"
    fi
else
    echo "Using model config from environment"
fi

echo "  Backend: http://localhost:8000"
echo "  Model context: ${MODEL_MAX_CONTEXT} tokens"
echo "  Tool format: ${MODEL_TOOL_FORMAT}"
echo "  Log file: $LOG_FILE (cleared)"
echo ""

if [[ -n "$DEBUG_FLAG" ]]; then
    echo "Starting proxy with debug enabled..."
    python3 compression_proxy.py $DEBUG_FLAG 2>&1 | tee -a "$LOG_FILE"
else
    python3 compression_proxy.py >> "$LOG_FILE" 2>&1
fi

