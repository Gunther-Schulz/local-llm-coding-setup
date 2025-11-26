#!/bin/bash
echo 'Stopping all servers...'

# Kill llama-cpp-python server (port 8000)
pkill -9 -f 'llama_cpp.server' && echo 'Stopped llama-cpp-python server' || echo 'No llama-cpp-python server running'

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

# Kill any remaining python processes that might be holding GPU memory
# Look for llama.cpp or python processes using significant GPU memory
PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)
if [ -n "$PIDS" ]; then
    echo "Killing GPU processes: $PIDS"
    for PID in $PIDS; do
        CMDLINE=$(ps -p $PID -o comm= 2>/dev/null)
        if [[ "$CMDLINE" == "python"* ]] || [[ "$CMDLINE" == *"llama"* ]]; then
            kill -9 $PID 2>/dev/null && echo "Killed GPU process $PID ($CMDLINE)" || true
        fi
    done
else
    echo 'No GPU processes found'
fi

# Wait for GPU memory to clear
sleep 3

# Show GPU status
if command -v nvidia-smi &> /dev/null; then
    echo ''
    echo 'GPU Status:'
    nvidia-smi --query-gpu=memory.used,memory.total --format=csv
fi

echo 'Done!'
