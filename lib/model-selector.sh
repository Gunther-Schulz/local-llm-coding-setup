#!/bin/bash
# Model selector library for vLLM server

MODELS_CONF="${MODELS_CONF:-$ROOT/models.conf}"

# Download a model (internal function)
# All output goes to stderr so it's visible when called from select_model_interactive
download_model_function() {
    local model_key="$1"
    
    echo "════════════════════════════════════════════════════════════════" >&2
    echo "  📥 Downloading: $model_key" >&2
    echo "════════════════════════════════════════════════════════════════" >&2
    
    # Get model config
    local config
    config=$(get_model_config "$model_key")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Model '$model_key' not found in config" >&2
        return 1
    fi
    
    IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url desc <<< "$config"
    
    local full_path="$ROOT/$path"
    local dir=$(dirname "$full_path")
    
    # Check if already exists AND is complete
    if [[ -f "$full_path" ]]; then
        # Check for .aria2 control file (aria2c incomplete download)
        if [[ -f "${full_path}.aria2" ]]; then
            echo "⚠️  Partial download detected (.aria2 control file found)" >&2
            echo "    Resuming download..." >&2
            echo "" >&2
            # Continue to download logic below (don't return)
        else
            # Basic size sanity check: GGUF models are never < 1GB
            local size_bytes=$(stat -c%s "$full_path" 2>/dev/null || echo "0")
            if [[ $size_bytes -lt 1073741824 ]]; then
                local size_mb=$((size_bytes / 1048576))
                echo "⚠️  File too small: ${size_mb}MB (GGUF models should be > 1GB)" >&2
                echo "    Re-downloading..." >&2
                echo "" >&2
                # Delete corrupt file
                rm -f "$full_path" >&2
                # Continue to download logic below (don't return)
            else
                local file_size_gb=$(du -b "$full_path" 2>/dev/null | awk '{printf "%.1f", $1/1024/1024/1024}')
                echo "✓ Model already downloaded: $full_path ($file_size_gb GB)" >&2
                return 0
            fi
        fi
    fi
    
    # Check if URL exists
    if [[ -z "$url" || "$url" == "none" ]]; then
        echo "ERROR: No download URL configured for '$model_key'" >&2
        echo "Please download manually from HuggingFace and place in: $full_path" >&2
        return 1
    fi
    
    # Create directory
    mkdir -p "$dir"
    
    # Download with best available tool
    echo "From: $url" >&2
    echo "To:   $full_path" >&2
    echo "" >&2
    
    # Method 1: aria2c (FASTEST)
    if command -v aria2c &> /dev/null; then
        # Check for partial download
        if [[ -f "${full_path}.aria2" ]]; then
            echo "🔄 Resuming partial download..." >&2
        else
            echo "🚀 Using aria2c (16 parallel connections)..." >&2
        fi
        
        aria2c \
            --continue=true \
            --max-connection-per-server=16 \
            --min-split-size=1M \
            --split=16 \
            --file-allocation=none \
            --console-log-level=notice \
            --summary-interval=1 \
            --dir="$dir" \
            --out="$(basename "$full_path")" \
            "$url" >&2
        
        # Clean up .aria2 control file on success
        [[ -f "${full_path}.aria2" ]] && rm -f "${full_path}.aria2" >&2
    # Method 2: HuggingFace CLI
    elif command -v huggingface-cli &> /dev/null; then
        echo "Using huggingface-cli..." >&2
        if [[ "$url" =~ huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$ ]]; then
            local repo="${BASH_REMATCH[1]}"
            local filename="${BASH_REMATCH[2]}"
            HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download \
                "$repo" "$filename" \
                --local-dir "$dir" \
                --local-dir-use-symlinks False >&2
            [[ -f "$dir/$filename" && "$dir/$filename" != "$full_path" ]] && mv "$dir/$filename" "$full_path" >&2
        fi
    # Method 3: wget
    elif command -v wget &> /dev/null; then
        echo "Using wget..." >&2
        wget --continue --show-progress "$url" -O "$full_path" >&2
    # Method 4: curl
    elif command -v curl &> /dev/null; then
        echo "Using curl..." >&2
        curl -L --continue-at - "$url" -o "$full_path" >&2
    else
        echo "ERROR: No download tool found (aria2c, huggingface-cli, wget, or curl)" >&2
        return 1
    fi
    
    if [[ $? -eq 0 && -f "$full_path" ]]; then
        echo "" >&2
        echo "✓ Successfully downloaded: $model_key" >&2
        return 0
    else
        echo "" >&2
        echo "✗ Download failed" >&2
        return 1
    fi
}

# Load available models from config
load_models() {
    local models=()
    while IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url desc || [[ -n "$key" ]]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        models+=("$key|$name|$path|$tokenizer|$ctx|$tool_parser|$tool_format|$url|$desc")
    done < "$MODELS_CONF"
    printf '%s\n' "${models[@]}"
}

# Get model config by key
get_model_config() {
    local search_key="$1"
    while IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url desc || [[ -n "$key" ]]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        if [[ "$key" == "$search_key" ]]; then
            echo "$key|$name|$path|$tokenizer|$ctx|$tool_parser|$tool_format|$url|$desc"
            return 0
        fi
    done < "$MODELS_CONF"
    return 1
}

# Show interactive menu and return selected model key
# Note: Menu display goes to stderr (&>2), only the selected key goes to stdout
select_model_interactive() {
    echo "" >&2
    echo "════════════════════════════════════════════════════════════════" >&2
    echo "                    📦 LLM Model Selection" >&2
    echo "════════════════════════════════════════════════════════════════" >&2
    echo "" >&2
    
    local models=()
    local i=1
    
    # Check if file exists
    if [[ ! -f "$MODELS_CONF" ]]; then
        echo "ERROR: models.conf not found at: $MODELS_CONF" >&2
        echo "ROOT=$ROOT" >&2
        return 1
    fi
    
    while IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url desc || [[ -n "$key" ]]; do
        # Skip comments and empty lines
        [[ "$key" =~ ^#.*$ ]] && continue
        [[ -z "$key" ]] && continue
        
        # Check if model file exists AND is complete
        local full_path="$ROOT/$path"
        local status="✓"
        local status_color="\033[32m"  # green
        
        if [[ ! -f "$full_path" ]]; then
            status="✗"
            status_color="\033[31m"  # red
        else
            # Check for .aria2 control file (aria2c incomplete download)
            if [[ -f "${full_path}.aria2" ]]; then
                status="⚠"
                status_color="\033[33m"  # yellow
                key="${key}_partial"
            else
                # Basic size sanity check: GGUF models are never < 1GB
                local size_bytes=$(stat -c%s "$full_path" 2>/dev/null || echo "0")
                if [[ $size_bytes -lt 1073741824 ]]; then
                    status="⚠"
                    status_color="\033[33m"  # yellow
                    key="${key}_partial"
                fi
            fi
        fi
        
        models+=("$key")
        
        echo -e "  ${status_color}[${i}]${status} ${name}\033[0m" >&2
        echo "      Context: ${ctx} tokens | Tool Format: ${tool_format}" >&2
        echo "      ${desc}" >&2
        echo "" >&2
        
        ((i++))
    done < "$MODELS_CONF"
    
    echo "════════════════════════════════════════════════════════════════" >&2
    echo "" >&2
    
    # Get user selection
    local selection
    while true; do
        read -p "Select model [1-${#models[@]}] or 'q' to quit: " selection </dev/tty
        
        if [[ "$selection" == "q" || "$selection" == "Q" ]]; then
            echo "Cancelled." >&2
            return 1
        fi
        
        if [[ "$selection" =~ ^[0-9]+$ ]] && [ "$selection" -ge 1 ] && [ "$selection" -le "${#models[@]}" ]; then
            local idx=$((selection - 1))
            local selected_key="${models[$idx]}"
            
            # Remove _partial suffix if present (from incomplete downloads)
            local clean_key="${selected_key/_partial/}"
            
            # Check if model file exists and is valid
            local config
            config=$(get_model_config "$clean_key")
            IFS='|' read -r key name path tokenizer ctx tool_parser tool_format url desc <<< "$config"
            local full_path="$ROOT/$path"
            
            # Check if file missing or incomplete
            local needs_download=false
            if [[ ! -f "$full_path" ]]; then
                needs_download=true
            else
                # Check for .aria2 control file (aria2c incomplete download)
                if [[ -f "${full_path}.aria2" ]]; then
                    needs_download=true
                    echo "" >&2
                    echo "⚠️  Partial download detected (.aria2 control file found)" >&2
                    echo "    Will resume download..." >&2
                else
                    # Basic size sanity check: GGUF models are never < 1GB
                    local size_bytes=$(stat -c%s "$full_path" 2>/dev/null || echo "0")
                    if [[ $size_bytes -lt 1073741824 ]]; then
                        needs_download=true
                        local size_mb=$((size_bytes / 1048576))
                        echo "" >&2
                        echo "⚠️  File too small: ${size_mb}MB (GGUF models should be > 1GB)" >&2
                        echo "    Will re-download..." >&2
                        # Delete corrupt file
                        rm -f "$full_path" >&2
                    fi
                fi
            fi
            
            if [[ "$needs_download" == "true" ]]; then
                echo "" >&2
                echo "⚠️  Model not downloaded: $name" >&2
                echo "" >&2
                read -p "Download now? [Y/n]: " download_choice </dev/tty
                
                if [[ ! "$download_choice" =~ ^[Nn]$ ]]; then
                    # Auto-download
                    echo "" >&2
                    if download_model_function "$clean_key"; then
                        echo "" >&2
                        echo "✅ Download complete! Model ready to use." >&2
                        echo "${clean_key}"  # This goes to stdout (captured)
                        return 0
                    else
                        echo "" >&2
                        echo "❌ Download failed. Please try manually:" >&2
                        echo "   ./download-model.sh" >&2
                        return 1
                    fi
                else
                    echo "Please select a downloaded model (marked with ✓)" >&2
                    echo "" >&2
                    continue
                fi
            fi
            
            echo "${selected_key}"  # This goes to stdout (captured)
            return 0
        fi
        
        echo "Invalid selection. Please enter a number between 1 and ${#models[@]}." >&2
    done
}

# Validate that model file exists
validate_model() {
    local model_path="$1"
    local model_key="$2"  # Optional model key for better error message
    
    if [[ ! -f "$model_path" ]]; then
        echo ""
        echo "════════════════════════════════════════════════════════════════"
        echo "  ❌ ERROR: Model file not found"
        echo "════════════════════════════════════════════════════════════════"
        echo "  Model: $model_key"
        echo "  Expected path: $model_path"
        echo ""
        echo "  To download this model, run:"
        echo "    ./download-model.sh"
        echo ""
        echo "  Or download manually and place in the path above."
        echo "════════════════════════════════════════════════════════════════"
        echo ""
        return 1
    fi
    return 0
}

# Export model configuration as environment variables
export_model_config() {
    local key="$1"
    
    local config
    config=$(get_model_config "$key")
    if [[ $? -ne 0 ]]; then
        echo "ERROR: Model '$key' not found in $MODELS_CONF"
        echo ""
        echo "Available models:"
        load_models | cut -d'|' -f1
        return 1
    fi
    
    IFS='|' read -r model_key model_name model_path tokenizer_id max_ctx tool_parser tool_format download_url description <<< "$config"
    
    local full_path="$ROOT/$model_path"
    
    # Export variables for vLLM
    export SELECTED_MODEL_KEY="$model_key"
    export SELECTED_MODEL_NAME="$model_name"
    export VLLM_GGUF_MODEL="$full_path"
    export VLLM_TOKENIZER_ID="$tokenizer_id"
    export VLLM_MAX_LEN="$max_ctx"
    export VLLM_TOOL_PARSER="$tool_parser"
    
    # Export variables for compression proxy
    export MODEL_TOOL_FORMAT="$tool_format"
    export MODEL_MAX_CONTEXT="$max_ctx"
    export MODEL_DOWNLOAD_URL="$download_url"
    
    return 0
}
