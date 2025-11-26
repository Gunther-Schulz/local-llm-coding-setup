#!/bin/bash
cd /workspace

./start-llama-server.sh &
LLAMA_PID=$!
sleep 5

./start-compression-proxy.sh &
PROXY_PID=$!

echo "Servers started:"
echo "  llama-cpp-python: PID $LLAMA_PID (port 8000)"
echo "  Compression proxy: PID $PROXY_PID (port 8002)"
echo ""
echo "To stop: kill $LLAMA_PID $PROXY_PID"

wait
