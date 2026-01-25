#!/bin/bash
# Show complete configuration (coding + vision models)

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/vision-manager.sh"
source "$ROOT/lib/model-selector.sh"

# Auto-migrate from old files
migrate_from_dotfiles 2>/dev/null

echo "════════════════════════════════════════════════════════════════"
echo "  Complete System Configuration"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Coding model (GPU)
CODING_MODEL=$(get_current_model)
CONTEXT_MODE=$(get_context_mode)

echo "🖥️  CODING MODEL (GPU via vLLM)"
echo "────────────────────────────────────────────────────────────────"
if [[ -n "$CODING_MODEL" ]]; then
    export_model_config "$CODING_MODEL" >/dev/null 2>&1
    echo "  Model:        $CODING_MODEL"
    echo "  Display:      $SELECTED_MODEL_NAME"
    echo "  Context:      $VLLM_MAX_LEN tokens"
    if [[ "$CONTEXT_MODE" == "extended" ]]; then
        echo "  Mode:         🟡 Extended (${MODEL_EXTENDED_CONTEXT} tokens)"
    else
        echo "  Mode:         🟢 Normal ($VLLM_MAX_LEN tokens)"
    fi
    echo "  Tool Parser:  $VLLM_TOOL_PARSER"
    echo "  Selected:     $(read_config "model" "selected_at")"
else
    echo "  ⚠️  No coding model selected"
    echo "  Run: ./select-model.sh"
fi

echo ""
echo "👁️  VISION MODEL (CPU via llama.cpp)"
echo "────────────────────────────────────────────────────────────────"
VISION_MODEL=$(read_config "vision" "model")
if [[ -n "$VISION_MODEL" ]]; then
    if export_vision_model_config "$VISION_MODEL" 2>/dev/null; then
        echo "  Model:        $VISION_MODEL"
        echo "  Display:      $VISION_MODEL_NAME"
        echo "  RAM Usage:    $VISION_RAM_USAGE"
        echo "  Context:      $VISION_MAX_CONTEXT tokens"
        echo "  Quantization: $VISION_QUANTIZATION"
        
        if [[ -f "$VISION_GGUF_PATH" && -f "$VISION_MMPROJ_PATH" ]]; then
            echo "  Status:       ✓ Downloaded"
        else
            echo "  Status:       ✗ Not downloaded"
            echo "  Download:     ./download-vision-model.sh $VISION_MODEL"
        fi
        echo "  Selected:     $(read_config "vision" "selected_at")"
    else
        echo "  ⚠️  Model config error: $VISION_MODEL"
    fi
else
    echo "  ⚠️  No vision model selected"
    echo "  Run: ./select-vision-model.sh"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📂 Configuration File: $ROOT/.llm-config"
echo ""
echo "Commands:"
echo "  Change coding model:  ./select-model.sh"
echo "  Change vision model:  ./select-vision-model.sh"
echo "  Toggle context mode:  ./toggle-context-mode.sh"
echo ""
