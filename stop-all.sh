#!/bin/bash
echo 'Stopping all servers...'

# Kill llama-cpp-python server (port 8000)
pkill -9 -f 'llama_cpp.server' && echo 'Stopped llama-cpp-python server' || echo 'No llama-cpp-python server running'

# Kill native llama-server (port 8000)
pkill -9 -f 'llama-server' && echo 'Stopped native llama-server' || echo 'No native llama-server running'

# Kill ALL vLLM-related processes (including EngineCore workers)
# First, kill the main vLLM API server
pkill -9 -f 'vllm.entrypoints.openai.api_server' && echo 'Stopped vLLM API server' || echo 'No vLLM API server running'

# Then kill ALL vLLM processes including workers and EngineCore
pkill -9 -f 'VLLM::' && echo 'Stopped vLLM worker processes' || echo 'No vLLM workers running'
pkill -9 -f 'vllm' && echo 'Stopped any remaining vLLM processes' || echo 'No remaining vLLM processes'

# Kill start script if still running
pkill -9 -f 'start-all-vllm.sh' && echo 'Stopped start script' || true
pkill -9 -f 'start-vllm-server.sh' && echo 'Stopped vLLM start script' || true

# Kill compression proxy (port 8002)
pkill -9 -f 'compression_proxy.py' && echo 'Stopped compression proxy' || echo 'No compression proxy running'

# Force kill any processes on ports 8000 and 8002
if command -v fuser &> /dev/null; then
    fuser -k 8000/tcp 2>/dev/null && echo 'Killed process on port 8000' || echo 'No process on port 8000'
    fuser -k 8002/tcp 2>/dev/null && echo 'Killed process on port 8002' || echo 'No process on port 8002'
else
    lsof -ti:8000 | xargs -r kill -9 2>/dev/null && echo 'Killed process on port 8000' || echo 'No process on port 8000'
    lsof -ti:8002 | xargs -r kill -9 2>/dev/null && echo 'Killed process on port 8002' || echo 'No process on port 8002'
fi

# Kill any remaining processes that might be holding GPU memory
# This is the nuclear option - kill ALL processes using the GPU except system ones
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -E '^[0-9]+$')
if [ -n "$PIDS" ]; then
    echo "Found GPU processes: $PIDS"
    for PID in $PIDS; do
        # Validate PID is numeric
        if ! [[ "$PID" =~ ^[0-9]+$ ]]; then
            continue
        fi
        
        CMDLINE=$(ps -p $PID -o args= 2>/dev/null | head -c 100)
        COMM=$(ps -p $PID -o comm= 2>/dev/null)
        
        # Skip if process doesn't exist
        if [ -z "$COMM" ]; then
            continue
        fi
        
        # Skip system processes
        if [[ "$COMM" == "gnome-shell" ]] || [[ "$COMM" == "Xwayland" ]] || [[ "$COMM" == "nautilus" ]]; then
            echo "Skipping system process $PID ($COMM)"
            continue
        fi
        
        # Kill anything that looks like our LLM servers
        if [[ "$CMDLINE" == *"python"* ]] || [[ "$CMDLINE" == *"llama"* ]] || [[ "$CMDLINE" == *"vllm"* ]] || [[ "$CMDLINE" == *"VLLM"* ]]; then
            echo "Killing GPU process $PID: $CMDLINE"
            kill -9 $PID 2>/dev/null && echo "  ✓ Killed $PID" || echo "  ✗ Failed to kill $PID"
        else
            echo "Unknown GPU process $PID: $CMDLINE (skipping)"
        fi
    done
else
    echo 'No GPU compute processes found (or nvidia-smi not available)'
fi

# Wait for GPU memory to clear
echo 'Waiting for GPU memory to clear...'
sleep 3

# Check if processes were actually killed
REMAINING=$(ps aux | grep -iE "(vllm|enginecore)" | grep -v grep | wc -l)
if [ "$REMAINING" -gt 0 ]; then
    echo "⚠ Warning: $REMAINING vLLM processes still running:"
    ps aux | grep -iE "(vllm|enginecore)" | grep -v grep
fi

# Show GPU status
if command -v nvidia-smi &> /dev/null; then
    echo ''
    echo '=== GPU Status ==='
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
    echo ''
    echo '=== GPU Compute Processes ==='
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi

echo ''
echo '✓ Done!'
