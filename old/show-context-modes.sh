#!/bin/bash
# Show available context modes for current model

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/model-selector.sh"

# Get current model
MODEL_KEY=$(get_current_model)
if [[ -z "$MODEL_KEY" ]]; then
    echo "No model selected. Run ./select-model.sh first."
    exit 1
fi

# Get model config
config=$(get_model_config "$MODEL_KEY")
if [[ -z "$config" ]]; then
    echo "Error: Model configuration not found"
    exit 1
fi

# config format: key|name|path|tokenizer|ctx|tool_parser|tool_format|url|ext_ctx|desc
IFS='|' read -r KEY NAME PATH TOKENIZER NORMAL_CTX TOOL_PARSER TOOL_FORMAT URL EXTENDED_CTX _rest <<< "$config"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  📊 Context Modes for $MODEL_KEY"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "🟢 NORMAL MODE (Recommended)"
echo "   Context:      $NORMAL_CTX tokens (~$(($NORMAL_CTX / 4))K chars)"
echo "   Performance:  40-60 tokens/sec"
echo "   VRAM:         GPU only"
echo "   Start with:   ./start-all-vllm.sh"
echo ""
echo "🟡 EXTENDED MODE (Slower)"
echo "   Context:      $EXTENDED_CTX tokens (~$(($EXTENDED_CTX / 4))K chars)"
echo "   Performance:  20-30 tokens/sec (50-70% slower)"
echo "   VRAM:         GPU + CPU offloading"
echo "   Start with:   ./start-extended-mode.sh"
echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "💡 Tips:"
echo "   • Normal mode is best for most coding tasks"
echo "   • Extended mode for large codebases (50K+ token contexts)"
echo "   • Consider tool-based context instead of extended mode"
echo ""
