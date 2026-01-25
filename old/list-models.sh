#!/bin/bash
# List all available models with their status

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
source "$ROOT/lib/model-selector.sh"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                    📦 Available Models"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Read current selection
CURRENT_MODEL_FILE="$ROOT/.current-model"
CURRENT_MODEL=""
if [[ -f "$CURRENT_MODEL_FILE" ]]; then
    CURRENT_MODEL=$(cat "$CURRENT_MODEL_FILE")
fi

# List all models (models.conf: key|name|path|tokenizer|ctx|tool_parser|tool_format|url|ext_ctx|desc)
while IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url ext_ctx desc || [[ -n "$key" ]]; do
    # Skip comments and empty lines
    [[ "$key" =~ ^#.*$ ]] && continue
    [[ -z "$key" ]] && continue
    
    full_path="$ROOT/$path"
    status="✗ Not downloaded"
    status_color="\033[31m"  # red
    current_marker=""
    
    # Check if this is the current model
    if [[ "$key" == "$CURRENT_MODEL" ]]; then
        current_marker=" 👉 CURRENT"
    fi
    
    if [[ -f "$full_path" ]]; then
        # Check for .aria2 control file (partial download)
        if [[ -f "${full_path}.aria2" ]]; then
            status="⚠ Partial download"
            status_color="\033[33m"  # yellow
        else
            # Basic size check
            size_bytes=$(stat -c%s "$full_path" 2>/dev/null || echo "0")
            if [[ $size_bytes -lt 1073741824 ]]; then
                size_mb=$((size_bytes / 1048576))
                status="⚠ Too small (${size_mb}MB)"
                status_color="\033[33m"  # yellow
            else
                file_size_gb=$(du -b "$full_path" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024/1024}')
                status="✓ Downloaded (${file_size_gb}GB)"
                status_color="\033[32m"  # green
            fi
        fi
    fi
    
    echo -e "  ${status_color}[$status]\033[0m ${key}${current_marker}"
    echo "      Name: $name"
    echo "      Context: ${ctx} tokens | Tool parser: ${tool_parser} | Tool format: ${tool_format}"
    if [[ "$url" != "none" ]]; then
        echo "      Download: Available"
    else
        echo "      Download: Manual only"
    fi
    echo "      Note: $desc"
    echo ""
done < "$MODELS_CONF"

echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To select a model:"
echo "  ./select-model.sh"
echo ""
echo "To download a model:"
echo "  ./download-model.sh"
echo ""
echo "To start servers:"
echo "  ./start-all-vllm.sh              # Use saved selection"
echo "  ./start-all-vllm.sh -m MODEL_KEY # Temporarily override"
echo ""
echo "════════════════════════════════════════════════════════════════"
