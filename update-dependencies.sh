#!/bin/bash
# Update all dependencies to latest versions

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda not found. Please install miniconda first."
    exit 1
fi

# Activate environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate glm4

echo "Updating llama-cpp-python with CUDA support..."
pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --no-cache-dir

echo "Updating other dependencies..."
pip install --upgrade 'jinja2>=3.1.0' 'fastapi>=0.109.0' 'uvicorn[standard]>=0.27.0' 'llmlingua>=0.2.1' 'sentence-transformers>=2.3.0' 'requests>=2.31.0' 'hf_transfer>=0.1.5' 'pydantic>=2.0.0' 'sse-starlette>=2.0.0' 'anyio>=4.0.0'

echo "Installed versions:"
pip list | grep -E "(jinja2|fastapi|uvicorn|llama-cpp-python|llmlingua|pydantic)"

echo ""
echo "Update complete! Restart servers with ./stop-all.sh && ./start-all.sh"


