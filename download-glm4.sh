#!/bin/bash
# Download GLM-4-32B-0414 AWQ model
# Run this on the RunPod instance

set -e

MODEL_NAME="AMead10/GLM-4-32B-0414-awq"
MODEL_DIR="/workspace/models/glm-4-32b-0414-awq"

echo "📥 Downloading GLM-4-32B-0414 AWQ model..."
echo "Model: $MODEL_NAME"
echo "Destination: $MODEL_DIR"
echo ""

# Activate conda environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate vllm-test

# Check if model already exists
if [ -d "$MODEL_DIR" ] && [ -f "$MODEL_DIR/config.json" ]; then
    echo "✅ Model already exists at $MODEL_DIR"
    echo "Skipping download. To re-download, delete the directory first."
    exit 0
fi

# Create model directory
mkdir -p "$MODEL_DIR"

# Install hf_transfer for faster downloads
if ! python -c "import hf_transfer" 2>/dev/null; then
    echo "Installing hf_transfer for faster downloads..."
    pip install -q hf_transfer
fi

# Download model
echo "Downloading model (this may take a while, ~20GB)..."
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR" || \
HF_HUB_ENABLE_HF_TRANSFER=0 huggingface-cli download "$MODEL_NAME" --local-dir "$MODEL_DIR"

# Verify download
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ Model download failed. Please check the error above."
    exit 1
fi

echo ""
echo "✅ Model downloaded successfully to $MODEL_DIR"
echo ""
echo "Model size:"
du -sh "$MODEL_DIR"


