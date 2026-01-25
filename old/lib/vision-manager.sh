#!/bin/bash
# Vision model management library

ROOT="${ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && cd .. && pwd)}"
VISION_MODELS_CONF="$ROOT/vision-models.conf"

# Get vision model configuration
get_vision_model_config() {
    local model_key="$1"
    
    if [[ ! -f "$VISION_MODELS_CONF" ]]; then
        echo "ERROR: vision-models.conf not found" >&2
        return 1
    fi
    
    grep "^${model_key}|" "$VISION_MODELS_CONF" | head -1
}

# Export vision model configuration
export_vision_model_config() {
    local model_key="$1"
    
    local config
    config=$(get_vision_model_config "$model_key")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Vision model '$model_key' not found" >&2
        return 1
    fi
    
    IFS='|' read -r key name gguf_path mmproj_path ctx quant url_model url_mmproj ram caps <<< "$config"
    
    # Export for scripts
    export VISION_MODEL_KEY="$key"
    export VISION_MODEL_NAME="$name"
    export VISION_GGUF_PATH="$ROOT/$gguf_path"
    export VISION_MMPROJ_PATH="$ROOT/$mmproj_path"
    export VISION_MAX_CONTEXT="$ctx"
    export VISION_QUANTIZATION="$quant"
    export VISION_RAM_USAGE="$ram"
    export VISION_CAPABILITIES="$caps"
    
    return 0
}

# Check if vision model is downloaded
is_vision_model_downloaded() {
    local model_key="$1"
    
    local config
    config=$(get_vision_model_config "$model_key")
    [[ $? -ne 0 ]] && return 1
    
    IFS='|' read -r key name gguf_path mmproj_path _ <<< "$config"
    
    [[ -f "$ROOT/$gguf_path" && -f "$ROOT/$mmproj_path" ]]
}
