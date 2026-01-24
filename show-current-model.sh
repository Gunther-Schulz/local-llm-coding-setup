#!/bin/bash
# Show currently selected model and its configuration

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
CURRENT_MODEL_FILE="$ROOT/.current-model"

if [[ ! -f "$CURRENT_MODEL_FILE" ]]; then
    echo ""
    echo "No model selected yet."
    echo ""
    echo "Run: ./select-model.sh"
    echo ""
    exit 1
fi

SELECTED_MODEL=$(cat "$CURRENT_MODEL_FILE")

# Load model selector library
source "$ROOT/lib/model-selector.sh"

# Get and export config
if export_model_config "$SELECTED_MODEL"; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  📦 Current Model Configuration"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "  Key:           $SELECTED_MODEL_KEY"
    echo "  Name:          $SELECTED_MODEL_NAME"
    echo "  Model file:    $VLLM_GGUF_MODEL"
    echo "  Tokenizer:     $VLLM_TOKENIZER_ID"
    echo "  Context:       $VLLM_MAX_LEN tokens"
    echo "  Tool parser:   $VLLM_TOOL_PARSER"
    echo "  Tool format:   $MODEL_TOOL_FORMAT"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "📝 Cursor IDE Configuration:"
    echo "  API Endpoint: http://localhost:8002/v1"
    echo "  Model Name:   $SELECTED_MODEL_KEY"
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "To change: ./select-model.sh"
    echo ""
else
    echo "Error: Could not load model configuration"
    exit 1
fi
