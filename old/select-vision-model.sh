#!/bin/bash
# Select vision model for CPU inference

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
export ROOT

source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/vision-manager.sh"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  👁️  Vision Model Selection (CPU-based)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check current selection
CURRENT_VISION_MODEL=$(read_config "vision" "model")
if [[ -n "$CURRENT_VISION_MODEL" ]]; then
    echo "Current vision model: $CURRENT_VISION_MODEL"
    echo ""
    read -p "Change vision model? [y/N]: " change_choice
    
    if [[ ! "$change_choice" =~ ^[Yy]$ ]]; then
        echo "Keeping current vision model: $CURRENT_VISION_MODEL"
        exit 0
    fi
    echo ""
fi

echo "Available Vision Models:"
echo ""

# List models
declare -a models=()
declare -a model_keys=()
i=1

while IFS='|' read -r key name gguf_path mmproj_path ctx quant url_model url_mmproj ram caps; do
    # Skip comments and empty lines
    [[ "$key" =~ ^#.*$ || -z "$key" ]] && continue
    
    model_keys+=("$key")
    models+=("$name")
    
    # Check if downloaded
    status="✗"
    if [[ -f "$ROOT/$gguf_path" && -f "$ROOT/$mmproj_path" ]]; then
        status="✓"
    fi
    
    echo "  [$i]$status $name"
    echo "      RAM: $ram | Context: $ctx tokens | Quant: $quant"
    echo "      $caps"
    echo ""
    
    ((i++))
done < "$ROOT/vision-models.conf"

echo "════════════════════════════════════════════════════════════════"
echo "Select vision model [1-$((i-1))] or 'q' to quit: "
read -r choice

if [[ "$choice" == "q" || "$choice" == "Q" ]]; then
    echo "Cancelled"
    exit 0
fi

if [[ ! "$choice" =~ ^[0-9]+$ ]] || [[ $choice -lt 1 ]] || [[ $choice -ge $i ]]; then
    echo "Invalid choice"
    exit 1
fi

SELECTED_KEY="${model_keys[$((choice-1))]}"
SELECTED_NAME="${models[$((choice-1))]}"

echo ""
echo "Selected: $SELECTED_NAME"
echo ""

# Check if downloaded
if ! is_vision_model_downloaded "$SELECTED_KEY"; then
    echo "⚠️  Model not downloaded: $SELECTED_NAME"
    echo ""
    read -p "Download now? [Y/n]: " download_choice
    
    if [[ ! "$download_choice" =~ ^[Nn]$ ]]; then
        echo ""
        ./download-vision-model.sh "$SELECTED_KEY"
    else
        echo "Model not downloaded. Download later with:"
        echo "  ./download-vision-model.sh $SELECTED_KEY"
    fi
fi

# Save selection
write_config "vision" "model" "$SELECTED_KEY"
write_config "vision" "selected_at" "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Vision Model Selected: $SELECTED_KEY"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Usage:"
echo "  Query an image:  ./query-vision.sh -i screenshot.png"
echo "  Change model:    ./select-vision-model.sh"
echo ""
echo "Note: Vision models run on CPU alongside your GPU coding model"
echo "════════════════════════════════════════════════════════════════"
