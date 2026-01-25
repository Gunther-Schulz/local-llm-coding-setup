#!/bin/bash
# Query vision model with an image

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/vision-manager.sh"

# Parse arguments
VISION_MODEL_KEY=""
IMAGE_PATH=""
PROMPT="Describe what you see in this image in detail."

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            VISION_MODEL_KEY="$2"
            shift 2
            ;;
        -i|--image)
            IMAGE_PATH="$2"
            shift 2
            ;;
        -p|--prompt)
            PROMPT="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -m, --model MODEL_KEY    Vision model to use"
            echo "  -i, --image PATH         Image file to analyze (required)"
            echo "  -p, --prompt TEXT        Prompt/question about the image"
            echo "  -h, --help              Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -i screenshot.png"
            echo "  $0 -i code.png -p 'What does this code do?'"
            echo "  $0 -m qwen2-vl-7b-q4 -i diagram.jpg -p 'Explain this architecture'"
            echo ""
            exit 0
            ;;
        *)
            # Assume it's an image path if no flag
            if [[ -z "$IMAGE_PATH" ]]; then
                IMAGE_PATH="$1"
            fi
            shift
            ;;
    esac
done

if [[ -z "$IMAGE_PATH" ]]; then
    echo "ERROR: Image path required"
    echo ""
    echo "Usage: $0 -i IMAGE_PATH [-p PROMPT]"
    echo "Or:    $0 IMAGE_PATH"
    echo ""
    exit 1
fi

if [[ ! -f "$IMAGE_PATH" ]]; then
    echo "ERROR: Image file not found: $IMAGE_PATH"
    exit 1
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

# Get vision model
if [[ -z "$VISION_MODEL_KEY" ]]; then
    VISION_MODEL_KEY=$(read_config "vision" "model")
    
    if [[ -z "$VISION_MODEL_KEY" ]]; then
        echo "ERROR: No vision model selected"
        echo ""
        echo "Run: ./select-vision-model.sh"
        echo "Or use: $0 -m MODEL_KEY -i IMAGE"
        echo ""
        exit 1
    fi
fi

# Load model config
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

echo "════════════════════════════════════════════════════════════════"
echo "  👁️  Vision Query"
echo "════════════════════════════════════════════════════════════════"
echo "  Model: $VISION_MODEL_NAME"
echo "  Image: $IMAGE_PATH"
echo "  Prompt: $PROMPT"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Processing... (this may take 5-15 seconds on CPU)"
echo ""

# Run vision query
"$LLAMACPP_BIN" \
    -m "$VISION_GGUF_PATH" \
    --mmproj "$VISION_MMPROJ_PATH" \
    -p "$PROMPT" \
    --image "$IMAGE_PATH" \
    -ngl 0 \
    -c "$VISION_MAX_CONTEXT" \
    --temp 0.7 \
    --top-p 0.9 \
    -n 512

echo ""
echo "════════════════════════════════════════════════════════════════"
