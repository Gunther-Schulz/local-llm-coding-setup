#!/bin/bash
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

# Parse command line arguments
SELECTED_MODEL=""
SHOW_HELP=false
ALLOW_FULL_CUDAGRAPH=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            SELECTED_MODEL="$2"
            shift 2
            ;;
        -f|--full-cudagraph)
            ALLOW_FULL_CUDAGRAPH=true
            shift
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
    echo "  -m, --model MODEL_KEY    Override model (see models.conf)"
    echo "  -f, --full-cudagraph    Use default FULL_AND_PIECEWISE (may freeze on RTX 5090+AMD)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                      # Use saved model from .llm-config (safe: PIECEWISE)"
    echo "  $0 -m qwen3-30b-q2      # Temporarily override with Qwen3-30B Q2_K"
    echo "  $0 -f                   # Test full cudagraphs (risky on 5090+AMD)"
    echo ""
    echo "To select/change model:"
    echo "  ./select-model.sh"
    echo ""
    exit 0
fi

# Load config manager and model selector library
source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/model-selector.sh"

# Determine which model to use (priority order):
# 1. Command line -m flag (temporary override)
# 2. Saved selection in .llm-config

if [[ -z "$SELECTED_MODEL" ]]; then
    # No -m flag provided, check for saved selection
    SELECTED_MODEL=$(get_current_model)
    
    if [[ -z "$SELECTED_MODEL" ]]; then
        echo ""
        echo "⚠️  No model selected!"
        echo ""
        echo "Please run: ./select-model.sh"
        echo "Or use:     $0 -m MODEL_KEY"
        echo ""
        exit 1
    fi
    
    echo ""
    echo "Using saved model: $SELECTED_MODEL"
    echo "(Change with: ./select-model.sh or override with: $0 -m MODEL_KEY)"
    echo ""
else
    echo ""
    echo "Using override model: $SELECTED_MODEL"
    echo "(Saved model will be used on next normal start)"
    echo ""
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

# Load context mode from centralized config
if [[ -z "$EXTENDED_CONTEXT_MODE" ]]; then
    EXTENDED_CONTEXT_MODE=$(get_extended_context_mode)
fi

echo "Starting vLLM server..."
echo "Logs: $LOG_FILE"
echo ""

# Check if extended context mode is enabled
ACTUAL_MAX_LEN="$VLLM_MAX_LEN"
if [[ "$EXTENDED_CONTEXT_MODE" == "1" ]]; then
    # Use extended context if available
    if [[ -n "$MODEL_EXTENDED_CONTEXT" && "$MODEL_EXTENDED_CONTEXT" != "$VLLM_MAX_LEN" ]]; then
        ACTUAL_MAX_LEN="$MODEL_EXTENDED_CONTEXT"
        SCALE_FACTOR=$(awk "BEGIN {printf \"%.1f\", $MODEL_EXTENDED_CONTEXT / $VLLM_MAX_LEN}")
        
        # IMPORTANT: Update MODEL_MAX_CONTEXT for compression proxy
        export MODEL_MAX_CONTEXT="$MODEL_EXTENDED_CONTEXT"
        
        echo "🟡 Extended context mode: $ACTUAL_MAX_LEN tokens (${SCALE_FACTOR}x with YaRN)"
        echo "   ⚠️  Performance will be 50-70% slower"
        echo ""
    else
        echo "⚠️  Extended mode requested but model doesn't support it"
        echo "   Using normal context: $ACTUAL_MAX_LEN tokens"
        echo ""
    fi
else
    echo "🟢 Normal context mode: $ACTUAL_MAX_LEN tokens"
    echo ""
fi

# Build vLLM command with model-specific tool parser
# By default use PIECEWISE cudagraph only (avoids "Capturing CUDA graphs (decode, FULL)"
# hard-freeze on RTX 5090 + AMD). Use -f/--full-cudagraph to test full cudagraphs.
VLLM_CMD="python -m vllm.entrypoints.openai.api_server \
  --model \"$VLLM_GGUF_MODEL\" \
  --tokenizer \"$VLLM_TOKENIZER_ID\" \
  --served-model-name \"$SELECTED_MODEL\" \
  --host \"$HOST\" \
  --port \"$PORT\" \
  --dtype \"$DTYPE\" \
  --max-model-len \"$ACTUAL_MAX_LEN\" \
  --tensor-parallel-size 1"
if [[ "$ALLOW_FULL_CUDAGRAPH" == true ]]; then
    echo "🔴 Full cudagraph mode (FULL_AND_PIECEWISE): may freeze on RTX 5090+AMD during decode capture."
    echo ""
else
    VLLM_CMD="$VLLM_CMD --compilation-config '{\"cudagraph_mode\": \"PIECEWISE\"}'"
fi

# Add YaRN scaling + CPU KV offload if extended mode (maximize context for dev work)
if [[ "$EXTENDED_CONTEXT_MODE" == "1" && -n "$SCALE_FACTOR" ]]; then
    # --cpu-offload-gb: offload KV cache to RAM when context exceeds VRAM (e.g. 50K–128K)
    # Use 8–16 GB if you have 32GB+ RAM; 4–8 if 16GB. Set 0 to disable.
    CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-8}"
    VLLM_CMD="$VLLM_CMD --rope-scaling '{\"type\":\"yarn\",\"factor\":${SCALE_FACTOR},\"original_max_position_embeddings\":${VLLM_MAX_LEN}}' --kv-cache-dtype fp8 --gpu-memory-utilization 0.85 --cpu-offload-gb $CPU_OFFLOAD_GB"
    echo "   CPU KV offload: ${CPU_OFFLOAD_GB} GB (set VLLM_CPU_OFFLOAD_GB to change)"
fi

# Add tool calling flags only if model supports it
if [[ "$VLLM_TOOL_PARSER" != "none" ]]; then
    VLLM_CMD="$VLLM_CMD --enable-auto-tool-choice --tool-call-parser \"$VLLM_TOOL_PARSER\""
    echo "Tool calling: ENABLED (parser: $VLLM_TOOL_PARSER)"
else
    echo "Tool calling: DISABLED (proxy will handle transformation)"
fi

echo ""

# Run vLLM. Use pipe through line-buffered tee so each line is written to the log
# immediately instead of sitting in a process buffer; on a crash, buffered output is lost.
# PYTHONUNBUFFERED=1 makes Python flush each print; stdbuf -oL makes tee flush each line.
PYTHONUNBUFFERED=1 eval "$VLLM_CMD" 2>&1 | stdbuf -oL tee -a "$LOG_FILE" >/dev/null
