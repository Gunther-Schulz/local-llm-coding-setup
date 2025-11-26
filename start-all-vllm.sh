#!/bin/bash
cd /workspace

./start-vllm-server.sh &
VLLM_PID=$!
sleep 5

./start-compression-proxy.sh &
PROXY_PID=$!

echo "Servers started:"
echo "  vLLM OpenAI server : PID $VLLM_PID (port 8000)"
echo "  Compression proxy  : PID $PROXY_PID (port 8002)"
echo ""
echo "To stop: kill $VLLM_PID $PROXY_PID    # or use ./stop-all.sh"

wait


