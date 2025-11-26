#!/bin/bash
# Start all services: vLLM server + compression proxy
# Run this on the RunPod instance

set -e

echo "🚀 Starting all services for GLM-4-32B-0414 AWQ..."

# Create logs directory
mkdir -p /workspace/logs

# Activate conda environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate vllm-test

# Start vLLM server in background
echo ""
echo "📡 Starting vLLM server on port 8000..."
cd /workspace/scripts
nohup bash start-vllm-glm4.sh > /workspace/logs/vllm.log 2>&1 &
VLLM_PID=$!
echo "vLLM server started (PID: $VLLM_PID)"
echo "Logs: /workspace/logs/vllm.log"

# Wait for vLLM to initialize
echo ""
echo "⏳ Waiting for vLLM to initialize (30 seconds)..."
sleep 30

# Check if vLLM is responding
if curl -s http://localhost:8000/v1/models > /dev/null 2>&1; then
    echo "✅ vLLM server is responding"
else
    echo "⚠️ vLLM server may not be ready yet, continuing anyway..."
fi

# Start compression proxy in background
echo ""
echo "🗜️ Starting compression proxy on port 8001..."
cd /workspace/scripts
VLLM_URL=http://127.0.0.1:8000/v1 PROXY_PORT=8001 MAX_CONTEXT_TOKENS=60000 \
nohup python compression_proxy.py > /workspace/logs/proxy.log 2>&1 &
PROXY_PID=$!
echo "Compression proxy started (PID: $PROXY_PID)"
echo "Logs: /workspace/logs/proxy.log"

# Wait a moment for proxy to start
sleep 5

# Check if proxy is responding
if curl -s http://localhost:8001/v1/models > /dev/null 2>&1; then
    echo "✅ Compression proxy is responding"
else
    echo "⚠️ Compression proxy may not be ready yet"
fi

echo ""
echo "✅ All services started!"
echo ""
echo "Services:"
echo "  - vLLM Server: http://0.0.0.0:8000 (direct)"
echo "  - Compression Proxy: http://0.0.0.0:8001 (recommended)"
echo ""
echo "Logs:"
echo "  - vLLM: tail -f /workspace/logs/vllm.log"
echo "  - Proxy: tail -f /workspace/logs/proxy.log"
echo ""
echo "To stop services:"
echo "  pkill -f 'vllm.entrypoints.openai.api_server'"
echo "  pkill -f 'compression_proxy.py'"
echo ""
echo "Or check PIDs:"
echo "  vLLM PID: $VLLM_PID"
echo "  Proxy PID: $PROXY_PID"


