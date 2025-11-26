#!/bin/bash
# Simple helper script to (re)download the Qwen2.5-Coder-14B-Instruct Q4_K_M GGUF model.
# Run this on the RunPod instance (inside /workspace) after setting up hf CLI.
# Uses Hugging Face's hf_transfer for faster downloads (via HF_HUB_ENABLE_HF_TRANSFER=1).
#
# Requires:
#   - `hf` CLI (`pip install -U 'huggingface_hub[cli,hf_transfer]'` or via conda)
#   - Network access from the pod

set -e

MODEL_REPO="yemiao2745/Qwen2.5-Coder-14B-Instruct-Q4_K_M-GGUF"
MODEL_FILE="qwen2.5-coder-14b-instruct-q4_k_m.gguf"
TARGET_DIR="/workspace/models/qwen2.5-coder-14b-q4_k_m"

echo "📥 Downloading Qwen2.5-Coder-14B-Instruct Q4_K_M GGUF..."
echo "  Repo : $MODEL_REPO"
echo "  File : $MODEL_FILE"
echo "  Dest : $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"

export HF_HUB_ENABLE_HF_TRANSFER=1

hf download "$MODEL_REPO" \
  --include "$MODEL_FILE" \
  --local-dir "$TARGET_DIR"

echo ""
echo "✅ Download complete."
echo "Model should now be at: $TARGET_DIR/$MODEL_FILE"


