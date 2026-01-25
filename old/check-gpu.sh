#!/bin/bash
# Quick GPU and vLLM process status checker

echo "=== vLLM Processes ==="
VLLM_PROCS=$(ps aux | grep -iE "(vllm|enginecore)" | grep -v grep)
if [ -z "$VLLM_PROCS" ]; then
    echo "✓ No vLLM processes running"
else
    echo "⚠ vLLM processes still running:"
    echo "$VLLM_PROCS"
fi

echo ""
echo "=== GPU Memory Usage ==="
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
    
    echo ""
    echo "=== GPU Compute Processes ==="
    GPU_PROCS=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null | grep -v '^$')
    if [ -z "$GPU_PROCS" ]; then
        echo "✓ No compute processes using GPU"
    else
        echo "$GPU_PROCS"
    fi
else
    echo "nvidia-smi not available"
fi

echo ""
echo "=== Ports 8000/8002 Status ==="
if command -v lsof &> /dev/null; then
    PORT_8000=$(lsof -ti:8000 2>/dev/null)
    PORT_8002=$(lsof -ti:8002 2>/dev/null)
    
    if [ -z "$PORT_8000" ]; then
        echo "✓ Port 8000: free"
    else
        echo "⚠ Port 8000: in use by PID $PORT_8000"
        ps -p $PORT_8000 -o pid,comm,args 2>/dev/null
    fi
    
    if [ -z "$PORT_8002" ]; then
        echo "✓ Port 8002: free"
    else
        echo "⚠ Port 8002: in use by PID $PORT_8002"
        ps -p $PORT_8002 -o pid,comm,args 2>/dev/null
    fi
else
    echo "lsof not available"
fi
