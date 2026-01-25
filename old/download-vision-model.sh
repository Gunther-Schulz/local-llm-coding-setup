#!/bin/bash
# Download vision models for llama.cpp

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

source "$ROOT/lib/config-manager.sh"

VISION_MODELS_CONF="$ROOT/vision-models.conf"

# Function to get vision model config
get_vision_model_config() {
    local model_key="$1"
    
    if [[ ! -f "$VISION_MODELS_CONF" ]]; then
        echo "ERROR: vision-models.conf not found" >&2
        return 1
    fi
    
    # Parse the config file
    grep "^${model_key}|" "$VISION_MODELS_CONF" | head -1
}

# Function to download vision model
download_vision_model() {
    local model_key="$1"
    
    echo "════════════════════════════════════════════════════════════════"
    echo "  📥 Downloading Vision Model: $model_key"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    # Get model config
    local config
    config=$(get_vision_model_config "$model_key")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Model '$model_key' not found in vision-models.conf" >&2
        return 1
    fi
    
    IFS='|' read -r key name gguf_path mmproj_path ctx quant url_model url_mmproj ram caps <<< "$config"
    
    local model_dir=$(dirname "$ROOT/$gguf_path")
    local mmproj_dir=$(dirname "$ROOT/$mmproj_path")
    
    # Create directories
    mkdir -p "$model_dir"
    mkdir -p "$mmproj_dir"
    
    local model_file="$ROOT/$gguf_path"
    local mmproj_file="$ROOT/$mmproj_path"
    
    # Download main model
    if [[ -f "$model_file" && ! -f "${model_file}.aria2" ]]; then
        local size=$(du -h "$model_file" | cut -f1)
        echo "✓ Model already downloaded: $model_file ($size)"
    else
        echo "📥 Downloading model file..."
        echo "  URL: $url_model"
        echo "  Destination: $model_file"
        echo ""
        
        if command -v aria2c &> /dev/null; then
            aria2c -x 8 -s 8 -k 1M -d "$(dirname "$model_file")" -o "$(basename "$model_file")" "$url_model"
        elif command -v wget &> /dev/null; then
            wget -c -O "$model_file" "$url_model"
        elif command -v curl &> /dev/null; then
            curl -L -C - -o "$model_file" "$url_model"
        else
            echo "ERROR: No download tool found (aria2c, wget, or curl required)" >&2
            return 1
        fi
        
        echo "✓ Model file downloaded"
    fi
    
    echo ""
    
    # Download mmproj model
    if [[ -f "$mmproj_file" && ! -f "${mmproj_file}.aria2" ]]; then
        local size=$(du -h "$mmproj_file" | cut -f1)
        echo "✓ Multimodal projection already downloaded: $mmproj_file ($size)"
    else
        echo "📥 Downloading multimodal projection..."
        echo "  URL: $url_mmproj"
        echo "  Destination: $mmproj_file"
        echo ""
        
        if command -v aria2c &> /dev/null; then
            aria2c -x 8 -s 8 -k 1M -d "$(dirname "$mmproj_file")" -o "$(basename "$mmproj_file")" "$url_mmproj"
        elif command -v wget &> /dev/null; then
            wget -c -O "$mmproj_file" "$url_mmproj"
        elif command -v curl &> /dev/null; then
            curl -L -C - -o "$mmproj_file" "$url_mmproj"
        else
            echo "ERROR: No download tool found (aria2c, wget, or curl required)" >&2
            return 1
        fi
        
        echo "✓ Multimodal projection downloaded"
    fi
    
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ Download Complete!"
    echo "════════════════════════════════════════════════════════════════"
    echo "  Model: $name"
    echo "  RAM usage: $ram"
    echo "  Capabilities: $caps"
    echo ""
    echo "Files:"
    echo "  Model: $model_file"
    echo "  MMProj: $mmproj_file"
    echo ""
}

# List available vision models
list_vision_models() {
    echo "════════════════════════════════════════════════════════════════"
    echo "  👁️  Available Vision Models (CPU-based)"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    
    local i=1
    while IFS='|' read -r key name gguf_path mmproj_path ctx quant url_model url_mmproj ram caps; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
        
        # Check if downloaded
        local status="✗"
        local model_file="$ROOT/$gguf_path"
        local mmproj_file="$ROOT/$mmproj_path"
        if [[ -f "$model_file" && -f "$mmproj_file" ]]; then
            status="✓"
        fi
        
        echo "  [$i]$status $name"
        echo "      Context: $ctx tokens | RAM: $ram | Quantization: $quant"
        echo "      $caps"
        echo ""
        
        ((i++))
    done < "$VISION_MODELS_CONF"
    
    echo "════════════════════════════════════════════════════════════════"
}

# Main script
if [[ $# -eq 0 ]]; then
    list_vision_models
    echo ""
    echo "Usage: $0 MODEL_KEY"
    echo ""
    echo "Example: $0 qwen2-vl-2b-q4"
    echo ""
    exit 0
fi

MODEL_KEY="$1"
download_vision_model "$MODEL_KEY"
