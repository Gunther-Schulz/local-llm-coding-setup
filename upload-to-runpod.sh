#!/bin/bash
# Upload setup scripts to RunPod instance
# Run this from your local machine

set -e

RUNPOD_HOST="149.36.0.117"
RUNPOD_PORT="11724"
RUNPOD_USER="root"
SSH_KEY="$HOME/.ssh/id_ed25519"

echo "📤 Uploading setup scripts to RunPod..."
echo "Host: $RUNPOD_USER@$RUNPOD_HOST:$RUNPOD_PORT"
echo ""

# Create scripts directory on RunPod
ssh -i $SSH_KEY -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST \
    "mkdir -p /workspace/scripts"

# Upload scripts
echo "Uploading scripts..."
scp -i $SSH_KEY -P $RUNPOD_PORT \
    setup-runpod-glm4.sh \
    download-glm4.sh \
    configure-yarn.sh \
    start-vllm-glm4.sh \
    start-compression-proxy.sh \
    start-all-services-glm4.sh \
    compression_proxy.py \
    $RUNPOD_USER@$RUNPOD_HOST:/workspace/scripts/

# Make scripts executable
echo ""
echo "Making scripts executable..."
ssh -i $SSH_KEY -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST \
    "chmod +x /workspace/scripts/*.sh"

echo ""
echo "✅ Upload complete!"
echo ""
echo "Next steps (SSH into RunPod):"
echo "  ssh -i $SSH_KEY -p $RUNPOD_PORT $RUNPOD_USER@$RUNPOD_HOST"
echo ""
echo "Then run:"
echo "  1. bash /workspace/scripts/setup-runpod-glm4.sh"
echo "  2. bash /workspace/scripts/download-glm4.sh"
echo "  3. bash /workspace/scripts/configure-yarn.sh"
echo "  4. bash /workspace/scripts/start-all-services-glm4.sh"

