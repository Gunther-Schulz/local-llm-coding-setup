#!/bin/bash
# Start llama.cpp vision server on CPU

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/vision-manager.sh"

# Parse arguments
VISION_MODEL_KEY=""
PORT=8003
SHOW_HELP=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            VISION_MODEL_KEY="$2"
            shift 2
            ;;
        -p|--port)
            PORT="$2"
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
    echo "  -m, --model MODEL_KEY    Vision model to use (from vision-models.conf)"
    echo "  -p, --port PORT          Port for vision server (default: 8003)"
    echo "  -h, --help              Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                              # Use saved vision model"
    echo "  $0 -m qwen2-vl-2b-q4            # Start with specific model"
    echo "  $0 -m qwen2-vl-7b-q4 -p 8004    # Custom port"
    echo ""
    exit 0
fi

# Check if llama.cpp is installed (llama-mtmd-cli from CMake build)
LLAMACPP_BIN="$ROOT/llama.cpp/build/bin/llama-mtmd-cli"
if [[ ! -f "$LLAMACPP_BIN" ]]; then
    echo "ERROR: llama.cpp not installed"
    echo ""
    echo "Run: ./setup-llamacpp.sh"
    echo ""
    exit 1
fi

# Get vision model from config or argument
if [[ -z "$VISION_MODEL_KEY" ]]; then
    VISION_MODEL_KEY=$(read_config "vision" "model")
    
    if [[ -z "$VISION_MODEL_KEY" ]]; then
        echo ""
        echo "⚠️  No vision model selected!"
        echo ""
        echo "Please run: ./select-vision-model.sh"
        echo "Or use:     $0 -m MODEL_KEY"
        echo ""
        exit 1
    fi
    
    echo ""
    echo "Using saved vision model: $VISION_MODEL_KEY"
    echo ""
else
    echo ""
    echo "Using specified vision model: $VISION_MODEL_KEY"
    echo ""
fi

# Load vision model config
if ! export_vision_model_config "$VISION_MODEL_KEY"; then
    exit 1
fi

# Check if model is downloaded
if [[ ! -f "$VISION_GGUF_PATH" || ! -f "$VISION_MMPROJ_PATH" ]]; then
    echo "ERROR: Vision model not downloaded"
    echo ""
    echo "Model file: $VISION_GGUF_PATH"
    echo "MMProj file: $VISION_MMPROJ_PATH"
    echo ""
    echo "Download with: ./download-vision-model.sh $VISION_MODEL_KEY"
    echo ""
    exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  👁️  Starting Vision Server (CPU-based)"
echo "════════════════════════════════════════════════════════════════"
echo "  Model:   $VISION_MODEL_NAME"
echo "  Port:    $PORT"
echo "  Context: $VISION_MAX_CONTEXT tokens"
echo "  RAM:     $VISION_RAM_USAGE"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "  Model file: $VISION_GGUF_PATH"
echo "  MMProj:     $VISION_MMPROJ_PATH"
echo ""

LOG_FILE="$ROOT/vision-server.log"
> "$LOG_FILE"

echo "Starting vision server..."
echo "Logs: $LOG_FILE"
echo ""
echo "⚠️  Note: First request will be slow as model loads into RAM"
echo ""

# Start llama.cpp server
# Note: llama.cpp doesn't have a built-in server mode for vision
# We'll create a simple wrapper or document direct CLI usage
echo "Vision server mode:"
echo "  This uses llama-qwen2vl-cli for single-shot inference"
echo "  For server mode, integration with llama-server is needed"
echo ""
echo "Direct usage example:"
echo "  $LLAMACPP_BIN \\"
echo "    -m \"$VISION_GGUF_PATH\" \\"
echo "    --mmproj \"$VISION_MMPROJ_PATH\" \\"
echo "    -p 'Describe this image' \\"
echo "    --image '/path/to/image.jpg' \\"
echo "    -ngl 0"
echo ""
echo "For now, use ./query-vision.sh to send vision queries"
echo ""
