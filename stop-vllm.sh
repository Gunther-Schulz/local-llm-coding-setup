#!/bin/bash
# Stop only the vLLM server

echo "Stopping vLLM server..."

# Kill main API server
pkill -9 -f 'vllm.entrypoints.openai.api_server' && echo "✓ Stopped vLLM API server" || echo "✗ No vLLM API server running"

# Kill worker processes
pkill -9 -f 'VLLM::' && echo "✓ Stopped vLLM worker processes" || echo "✗ No vLLM workers running"

# Kill any remaining vllm processes
pkill -9 -f 'vllm' && echo "✓ Stopped remaining vLLM processes" || echo "✗ No remaining vLLM processes"

# Check if port 8000 is free
if lsof -ti:8000 >/dev/null 2>&1; then
    echo "⚠ Port 8000 still in use:"
    lsof -i:8000
else
    echo "✓ Port 8000 is free"
fi

# Check GPU memory
echo ""
echo "GPU Status:"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null || echo "No GPU processes"
