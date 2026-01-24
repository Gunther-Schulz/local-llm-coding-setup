#!/bin/bash
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

# Parse command line arguments
SELECTED_MODEL=""
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            SELECTED_MODEL="$2"
            shift 2
            ;;
        -h|--help)
            SHOW_HELP=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            SHOW_HELP=true
            shift
            ;;
    esac
done

if [[ "$SHOW_HELP" == true ]]; then
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model MODEL_KEY    Select model by key (see models.conf)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Interactive model selection menu"
    echo "  $0 -m qwen3-30b-q2      # Load Qwen3-30B Q2_K directly"
    echo "  $0 -m qwen2.5-14b-q4    # Load current Qwen2.5-14B"
    echo ""
    exit 0
fi

# Load model selector library
source "$ROOT/lib/model-selector.sh"

# Determine which model to use (priority order):
# 1. Command line -m flag (highest priority)
# 2. Saved selection in .current-model
# 3. Interactive menu (fallback)

CURRENT_MODEL_FILE="$ROOT/.current-model"

if [[ -z "$SELECTED_MODEL" ]]; then
    # No -m flag provided, check for saved selection
    if [[ -f "$CURRENT_MODEL_FILE" ]]; then
        SELECTED_MODEL=$(cat "$CURRENT_MODEL_FILE")
        echo ""
        echo "Using saved model selection: $SELECTED_MODEL"
        echo "(Override with: $0 -m MODEL_KEY or run ./select-model.sh to change)"
        echo ""
    else
        # No saved selection, show interactive menu
        echo ""
        echo "No model selected yet. Please choose a model:"
        echo "(Tip: Run ./select-model.sh first to save your choice)"
        echo ""
        SELECTED_MODEL=$(select_model_interactive)
        if [[ $? -ne 0 || -z "$SELECTED_MODEL" ]]; then
            echo "No model selected. Exiting."
            exit 1
        fi
        # Save the selection for future use
        echo "$SELECTED_MODEL" > "$CURRENT_MODEL_FILE"
        echo ""
        echo "Model selection saved. Future runs will use this model automatically."
        echo ""
    fi
fi

# Export model configuration
if ! export_model_config "$SELECTED_MODEL"; then
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  🚀 Starting vLLM OpenAI-compatible Server"
echo "════════════════════════════════════════════════════════════════"
echo "  Model       : $SELECTED_MODEL_NAME"
echo "  GGUF file   : $VLLM_GGUF_MODEL"
echo "  Tokenizer   : $VLLM_TOKENIZER_ID"
echo "  Max context : $VLLM_MAX_LEN tokens"
echo "  Tool parser : ${VLLM_TOOL_PARSER}"
echo "  Tool format : ${MODEL_TOOL_FORMAT}"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Additional vLLM settings
DTYPE="${VLLM_DTYPE:-float16}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

cd "$ROOT"

# Activate conda environment
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
  source "$_conda_sh"
  conda activate llm
elif [ -n "$VIRTUAL_ENV" ]; then
  : # already in a venv
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  source "$ROOT/venv/bin/activate"
else
  echo "ERROR: No conda (env llm) or venv found. Create one:"
  echo "  conda create -n llm python=3.10 -y && conda activate llm"
  echo "  or: python3 -m venv $ROOT/.venv && . $ROOT/.venv/bin/activate"
  echo "Then: ./install-deps.sh   and   ./setup-vllm.sh"
  exit 1
fi

LOG_FILE="$ROOT/vllm-server.log"

# Clear log file on start
> "$LOG_FILE"

echo "Starting vLLM server..."
echo "Logs: $LOG_FILE"
echo ""

# Build vLLM command with model-specific tool parser
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
  --model \"$VLLM_GGUF_MODEL\" \
  --tokenizer \"$VLLM_TOKENIZER_ID\" \
  --served-model-name \"$SELECTED_MODEL_KEY\" \
  --host \"$HOST\" \
  --port \"$PORT\" \
  --dtype \"$DTYPE\" \
  --max-model-len \"$VLLM_MAX_LEN\" \
  --tensor-parallel-size 1"

# Add tool calling flags only if model supports it
if [[ "$VLLM_TOOL_PARSER" != "none" ]]; then
    VLLM_CMD="$VLLM_CMD --enable-auto-tool-choice --tool-call-parser \"$VLLM_TOOL_PARSER\""
    echo "Tool calling: ENABLED (parser: $VLLM_TOOL_PARSER)"
else
    echo "Tool calling: DISABLED (proxy will handle transformation)"
fi

echo ""

# Run vLLM
eval "$VLLM_CMD" >> "$LOG_FILE" 2>&1
