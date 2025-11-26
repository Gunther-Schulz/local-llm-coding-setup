#!/bin/bash
# Upload setup scripts to RunPod instance
# Run this from your local machine

set -e

RUNPOD_HOST="149.36.0.117"
RUNPOD_PORT="12844"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "📤 Uploading setup scripts to RunPod..."
echo "Host: $RUNPOD_USER@$RUNPOD_HOST:$RUNPOD_PORT"
echo ""

# Upload scripts directly to /workspace
echo "Uploading scripts..."
scp -i $SSH_KEY -P $RUNPOD_PORT \
    build-native-llama.sh \
    start-llama-server-native.sh \
    start-compression-proxy.sh \
    start-all-native.sh \
    stop-all.sh \
    compression_proxy.py \
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
echo "  ./build-native-llama.sh          # Build native llama.cpp (first time only)"
echo "  ./start-all-native.sh            # Start native llama-server + proxy"
echo ""
echo "Or use llama-cpp-python instead:"
echo "  ./start-all.sh                   # Start llama-cpp-python + proxy"

