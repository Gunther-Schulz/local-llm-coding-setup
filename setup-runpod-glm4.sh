#!/bin/bash
# Setup script for RunPod RTX 5090 with GLM-4-32B-0414 AWQ
# Run this on the RunPod instance via SSH
# Usage: bash setup-runpod-glm4.sh

set -e

echo "🚀 Setting up GLM-4-32B-0414 AWQ on RunPod RTX 5090..."
echo "Using /workspace for all persistent storage"

# Check GPU
echo ""
echo "📊 Checking GPU..."
nvidia-smi --query-gpu=name,memory.total --format=csv

# Create directory structure in /workspace
echo ""
echo "📁 Creating directory structure..."
mkdir -p /workspace/models
mkdir -p /workspace/scripts
mkdir -p /workspace/logs

# Install miniconda in /workspace if not present
if [ ! -d "/workspace/miniconda3" ]; then
    echo ""
    echo "📦 Installing Miniconda to /workspace..."
    cd /workspace
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    bash miniconda.sh -b -p /workspace/miniconda3
    rm miniconda.sh
    echo "✅ Miniconda installed to /workspace/miniconda3"
else
    echo "✅ Miniconda already exists at /workspace/miniconda3"
fi

# Initialize conda
export PATH="/workspace/miniconda3/bin:$PATH"
source /workspace/miniconda3/etc/profile.d/conda.sh

# Accept Conda Terms of Service
echo ""
echo "📝 Accepting Conda Terms of Service..."
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main || true
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r || true

# Create conda environment
echo ""
echo "🐍 Creating conda environment..."
if conda env list | grep -q "vllm-test"; then
    echo "✅ Conda environment 'vllm-test' already exists"
else
    conda create -n vllm-test python=3.10 -y
    echo "✅ Conda environment created"
fi

# Activate environment
conda activate vllm-test

# Install vLLM and dependencies
echo ""
echo "📦 Installing vLLM and dependencies..."
pip install --upgrade pip
pip install vllm huggingface-hub

# Install compression proxy dependencies
echo ""
echo "📦 Installing compression proxy dependencies..."
pip install llmlingua sentence-transformers flask requests scikit-learn

# Verify installation
echo ""
echo "✅ Verifying installation..."
python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download model: bash /workspace/scripts/download-glm4.sh"
echo "2. Configure YaRN: bash /workspace/scripts/configure-yarn.sh"
echo "3. Start services: bash /workspace/scripts/start-all-services.sh"


