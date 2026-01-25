#!/bin/bash
# Download models for the multi-model setup

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

# Load model selector library
source "$ROOT/lib/model-selector.sh"

# Legacy/override: use when models.conf download_url is "none" or unset.
# Should match models.conf download_url when both exist.
declare -A MODEL_URLS=(
    ["qwen3-30b-q2"]="https://huggingface.co/mradermacher/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct.Q2_K.gguf"
    ["qwen3-30b-q3_k_s"]="https://huggingface.co/mradermacher/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct.Q3_K_S.gguf"
    ["qwen3-30b-q3_k_m"]="https://huggingface.co/mradermacher/Qwen3-Coder-30B-A3B-Instruct-GGUF/resolve/main/Qwen3-Coder-30B-A3B-Instruct.Q3_K_M.gguf"
)

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "                    📥 Model Downloader"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Show available models
echo "Available models to download:"
echo ""
i=1
models=()

while IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url ext_ctx desc || [[ -n "$key" ]]; do
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    
    full_path="$ROOT/$path"
    status="✓ Downloaded"
    action="skip"
    
    if [[ ! -f "$full_path" ]]; then
        status="✗ Not downloaded"
        action="download"
    fi
    
    models+=("$key")
    
    echo "  [$i] $status - $name"
    echo "      Size: $(echo "$desc" | grep -oP '\d+\.\d+GB' || echo "See HuggingFace")"
    echo "      Context: ${ctx} tokens | Tool Format: ${tool_format}"
    echo ""
    
    ((i++))
done < "$MODELS_CONF"

echo "════════════════════════════════════════════════════════════════"
echo ""

# Get user selection
read -p "Select model to download [1-${#models[@]}], 'a' for all, or 'q' to quit: " selection

if [[ "$selection" == "q" || "$selection" == "Q" ]]; then
    echo "Cancelled."
    exit 0
fi

# Download function
download_model() {
    local model_key="$1"
    
    echo ""
    echo "Downloading: $model_key"
    echo "════════════════════════════════════════════════════════════════"
    
    # Get model config
    local config
    config=$(get_model_config "$model_key")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Model '$model_key' not found"
        return 1
    fi
    
    IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url ext_ctx desc <<< "$config"
    
    local full_path="$ROOT/$path"
    local dir=$(dirname "$full_path")
    
    # Check if already exists
    if [[ -f "$full_path" ]]; then
        echo "✓ Model already downloaded: $full_path"
        return 0
    fi
    
    # URL: prefer models.conf (download_url); fallback to MODEL_URLS for legacy/override
    local the_url=""
    if [[ -n "$url" && "$url" != "none" ]]; then
        the_url="$url"
    elif [[ -n "${MODEL_URLS[$model_key]:-}" ]]; then
        the_url="${MODEL_URLS[$model_key]}"
    fi
    if [[ -z "$the_url" ]]; then
        echo "ERROR: No download URL configured for '$model_key'"
        echo "Please download manually from HuggingFace and place in: $full_path"
        return 1
    fi
    
    # Create directory
    mkdir -p "$dir"
    
    # Download with best available tool
    local url="$the_url"
    echo "Downloading from: $url"
    echo "Destination: $full_path"
    echo ""
    
    # Method 1: aria2c (FASTEST - multi-connection, parallel downloads)
    if command -v aria2c &> /dev/null; then
        echo "🚀 Using aria2c (multi-connection download)..."
        aria2c \
            --continue=true \
            --max-connection-per-server=16 \
            --min-split-size=1M \
            --split=16 \
            --file-allocation=none \
            --console-log-level=warn \
            --summary-interval=0 \
            --dir="$dir" \
            --out="$(basename "$full_path")" \
            "$url"
    # Method 2: HuggingFace CLI with hf_transfer (fast, multi-threaded)
    elif command -v huggingface-cli &> /dev/null; then
        echo "Using huggingface-cli (with hf_transfer for speed)..."
        
        # Extract repo and filename from URL
        # URL format: https://huggingface.co/USER/REPO/resolve/main/FILE.gguf
        if [[ "$url" =~ huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$ ]]; then
            local repo="${BASH_REMATCH[1]}"
            local filename="${BASH_REMATCH[2]}"
            
            # Use hf_transfer for faster downloads (multi-threaded)
            HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
                "$repo" \
                "$filename" \
                --local-dir "$dir" \
                --local-dir-use-symlinks False
                
            # Move file if huggingface-cli put it in a subdirectory
            if [[ -f "$dir/$filename" && "$dir/$filename" != "$full_path" ]]; then
                mv "$dir/$filename" "$full_path"
            fi
        else
            echo "Warning: URL format not recognized for huggingface-cli, falling back..."
        fi
    # Method 3: wget (slower, but reliable)
    elif command -v wget &> /dev/null; then
        echo "Using wget (single-threaded)..."
        wget --continue --show-progress "$url" -O "$full_path"
    # Method 4: curl (slowest)
    elif command -v curl &> /dev/null; then
        echo "Using curl (single-threaded)..."
        curl -L --continue-at - "$url" -o "$full_path"
    else
        echo "ERROR: No download tool found"
        echo ""
        echo "Install aria2: sudo pacman -S aria2"
        echo "Or: pip install -U huggingface_hub[cli] hf-transfer"
        return 1
    fi
    
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✓ Successfully downloaded: $model_key"
        return 0
    else
        echo "✗ Download failed"
        return 1
    fi
}

# Download selected model(s)
if [[ "$selection" == "a" || "$selection" == "A" ]]; then
    echo "Downloading all missing models..."
    for model_key in "${models[@]}"; do
        download_model "$model_key"
    done
elif [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#models[@]}" ]; then
    local idx=$((selection - 1))
    download_model "${models[$idx]}"
else
    echo "Invalid selection."
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "Done! Use './start-vllm-server.sh' to select and run a model."
echo "════════════════════════════════════════════════════════════════"
