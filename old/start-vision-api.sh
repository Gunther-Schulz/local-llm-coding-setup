#!/bin/bash
# Start Vision API server

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/vision-manager.sh"

# Parse arguments
DEBUG_FLAG=""
PORT=8004

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--debug)
            DEBUG_FLAG="--debug"
            export DEBUG=1
            shift
            ;;
        -p|--port)
            PORT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-d|--debug] [-p|--port PORT]"
            exit 1
            ;;
    esac
done

# Check if llama.cpp is installed (llama-mtmd-cli from CMake build)
LLAMACPP_BIN="$ROOT/llama.cpp/build/bin/llama-mtmd-cli"
if [[ ! -f "$LLAMACPP_BIN" ]]; then
    echo "ERROR: llama.cpp not installed"
    echo ""
    echo "Run: ./setup-llamacpp.sh"
    echo ""
    exit 1
fi

# Get vision model
VISION_MODEL_KEY=$(read_config "vision" "model")

if [[ -z "$VISION_MODEL_KEY" ]]; then
    echo ""
    echo "⚠️  No vision model selected!"
    echo ""
    echo "Please run: ./select-vision-model.sh"
    echo ""
    exit 1
fi

# Load vision model config
if ! export_vision_model_config "$VISION_MODEL_KEY"; then
    exit 1
fi

# Check if model is downloaded
if [[ ! -f "$VISION_GGUF_PATH" || ! -f "$VISION_MMPROJ_PATH" ]]; then
    echo "ERROR: Vision model not downloaded"
    echo ""
    echo "Download with: ./download-vision-model.sh $VISION_MODEL_KEY"
    echo ""
    exit 1
fi

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
  :
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  source "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  source "$ROOT/venv/bin/activate"
else
  echo "ERROR: No conda (env llm) or venv. Create one and run ./install-deps.sh first."
  exit 1
fi

LOG_FILE="$ROOT/vision-api.log"
> "$LOG_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  👁️  Starting Vision API Server"
echo "════════════════════════════════════════════════════════════════"
echo "  Model:   $VISION_MODEL_NAME"
echo "  Port:    $PORT"
echo "  RAM:     $VISION_RAM_USAGE"
echo "  Context: $VISION_MAX_CONTEXT tokens"
echo "════════════════════════════════════════════════════════════════"
echo "  Model file: $VISION_GGUF_PATH"
echo "  MMProj:     $VISION_MMPROJ_PATH"
echo "  Log file:   $LOG_FILE"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Export for Python script
export VISION_GGUF_PATH
export VISION_MMPROJ_PATH
export VISION_MAX_CONTEXT
export LLAMACPP_BIN

if [[ -n "$DEBUG_FLAG" ]]; then
    echo "Starting vision API in DEBUG mode..."
    python3 vision-api-server.py --port "$PORT" $DEBUG_FLAG 2>&1 | tee -a "$LOG_FILE"
else
    echo "Starting vision API..."
    python3 vision-api-server.py --port "$PORT" >> "$LOG_FILE" 2>&1
fi
