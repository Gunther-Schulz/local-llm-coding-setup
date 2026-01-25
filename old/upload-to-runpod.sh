#!/bin/bash
# Upload setup scripts to RunPod instance
# Run this from your local machine

set -e

RUNPOD_HOST="149.36.1.181"
RUNPOD_PORT="42717"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "📤 Uploading setup scripts to RunPod..."
echo "Host: $RUNPOD_USER@$RUNPOD_HOST:$RUNPOD_PORT"
echo ""

# Upload scripts directly to /workspace
echo "Uploading scripts..."
scp -i $SSH_KEY -P $RUNPOD_PORT \
    build-native-llama.sh \
    install-deps.sh \
    download-qwen2.5-coder-14b.sh \
    start-llama-server-native.sh \
    start-compression-proxy.sh \
    start-all-native.sh \
    start-all-vllm.sh \
    setup-vllm.sh \
    start-vllm-server.sh \
    stop-all.sh \
    compression_proxy.py \
    requirements.txt \
    $RUNPOD_USER@$RUNPOD_HOST:/workspace/

# Make scripts executable
echo ""
echo "Making scripts executable..."
ssh -i $SSH_KEY -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST \
    "chmod +x /workspace/*.sh"

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps (SSH into RunPod):"
echo "  ssh -i $SSH_KEY -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST"
echo ""
echo "Then run:"
echo "  cd /workspace"
echo "  ./stop-all.sh                    # Stop any running servers"
echo "  ./install-deps.sh                # Install Python dependencies (first time only)"
echo "  ./build-native-llama.sh          # Build native llama.cpp (first time only)"
echo "  ./start-all-native.sh            # Start native llama-server + proxy"
echo ""
echo "Or use vLLM instead:"
echo "  ./download-qwen2.5-coder-14b.sh  # (Re)download Qwen2.5-Coder-14B GGUF if needed"
echo "  ./start-all-vllm.sh              # Start vLLM OpenAI server + proxy"

