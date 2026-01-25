#!/bin/bash
# Standalone model selector - saves selection for server scripts to use

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

# Export ROOT so library can use it
export ROOT

# Load libraries
source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/model-selector.sh"

# Auto-migrate from old dot-files if they exist
migrate_from_dotfiles

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                    📦 Model Selection"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if there's a current selection
CURRENT_MODEL_FILE="$ROOT/.current-model"
if [[ -f "$CURRENT_MODEL_FILE" ]]; then
    CURRENT_MODEL=$(cat "$CURRENT_MODEL_FILE")
    echo "Current model: $CURRENT_MODEL"
    echo ""
    read -p "Change model? [y/N]: " change_choice
    
    if [[ ! "$change_choice" =~ ^[Yy]$ ]]; then
        echo "Keeping current model: $CURRENT_MODEL"
        exit 0
    fi
    echo ""
fi

# Interactive model selection
SELECTED_MODEL=$(select_model_interactive)
if [[ $? -ne 0 || -z "$SELECTED_MODEL" ]]; then
    echo "No model selected."
    exit 1
fi

# Validate model exists
if ! export_model_config "$SELECTED_MODEL"; then
    exit 1
fi

# Save selection
echo "$SELECTED_MODEL" > "$CURRENT_MODEL_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Model Selected: $SELECTED_MODEL"
echo "════════════════════════════════════════════════════════════════"
echo "  Display name: $SELECTED_MODEL_NAME"
echo "  Context: $VLLM_MAX_LEN tokens"
echo "  Tool parser: $VLLM_TOOL_PARSER"
echo "  Tool format: $MODEL_TOOL_FORMAT"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📝 Cursor IDE Configuration:"
echo "  API Endpoint: http://localhost:8002/v1"
echo "  Model Name:   $SELECTED_MODEL"
echo ""
echo "This model will be used by:"
echo "  • ./start-vllm-server.sh"
echo "  • ./start-all-vllm.sh"
echo "  • ./start-compression-proxy.sh"
echo ""
echo "To change model later, run: ./select-model.sh"
echo "To override temporarily, use: ./start-all-vllm.sh -m MODEL_KEY"
echo "════════════════════════════════════════════════════════════════"
