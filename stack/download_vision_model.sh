#!/bin/bash
# Download vision model files (GGUF + MMProj)
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_KEY="$1"

if [ -z "$MODEL_KEY" ]; then
    echo "Usage: $0 <model_key>"
    echo ""
    echo "Available models:"
    grep -v '^#' config/vision-models.conf | grep -v '^$' | cut -d'|' -f1,2 | sed 's/|/ - /'
    echo ""
    exit 1
fi

# Parse model config
MODEL_LINE=$(grep "^${MODEL_KEY}|" config/vision-models.conf)

if [ -z "$MODEL_LINE" ]; then
    echo "Error: Model '$MODEL_KEY' not found in config/vision-models.conf"
    exit 1
fi

# Extract fields
IFS='|' read -r key name gguf_path mmproj_path max_ctx quant url_model url_mmproj ram caps <<< "$MODEL_LINE"

echo "=========================================="
echo "  Downloading Vision Model"
echo "=========================================="
echo "  Model: $name"
echo "  Quant: $quant"
echo "  RAM:   $ram"
echo "=========================================="
echo ""

# Create model directory
MODEL_DIR=$(dirname "$gguf_path")
mkdir -p "$MODEL_DIR"

# Download GGUF model
if [ -f "$gguf_path" ]; then
    echo "✓ GGUF model already exists: $gguf_path"
else
    echo "Downloading GGUF model..."
    echo "  From: $url_model"
    echo "  To:   $gguf_path"
    echo ""
    
    # Try aria2c first (fastest), fall back to curl
    if command -v aria2c &>/dev/null; then
        aria2c -x 8 -s 8 -k 1M -o "$gguf_path" "$url_model"
    elif command -v curl &>/dev/null; then
        curl -L -o "$gguf_path" "$url_model" --progress-bar
    else
        echo "Error: Neither aria2c nor curl found. Install one of them."
        exit 1
    fi
    
    echo "✓ GGUF model downloaded"
fi

# Download MMProj file
if [ -f "$mmproj_path" ]; then
    echo "✓ MMProj file already exists: $mmproj_path"
else
    echo ""
    echo "Downloading MMProj file..."
    echo "  From: $url_mmproj"
    echo "  To:   $mmproj_path"
    echo ""
    
    if command -v aria2c &>/dev/null; then
        aria2c -x 8 -s 8 -k 1M -o "$mmproj_path" "$url_mmproj"
    elif command -v curl &>/dev/null; then
        curl -L -o "$mmproj_path" "$url_mmproj" --progress-bar
    else
        echo "Error: Neither aria2c nor curl found. Install one of them."
        exit 1
    fi
    
    echo "✓ MMProj file downloaded"
fi

echo ""
echo "=========================================="
echo "  ✓ Vision Model Ready!"
echo "=========================================="
echo "  Model:  $name"
echo "  GGUF:   $(basename "$gguf_path")"
echo "  MMProj: $(basename "$mmproj_path")"
echo ""
echo "Select this model:"
echo "  ./run/select_vision_model.sh"
echo ""
echo "Start vision server:"
echo "  ./run/vision.sh"
echo "=========================================="
