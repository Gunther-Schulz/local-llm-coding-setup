#!/bin/bash
# Start vLLM + Compression Proxy + Vision API (complete setup)

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

# Parse arguments
MODEL_ARG=""
MODEL_KEY=""
DEBUG_FLAG=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--model)
            MODEL_ARG="-m $2"
            MODEL_KEY="$2"
            shift 2
            ;;
        -d|--debug)
            DEBUG_FLAG="--debug"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [-m MODEL_KEY] [-d|--debug]"
            exit 1
            ;;
    esac
done

echo "════════════════════════════════════════════════════════════════"
echo "  🚀 Starting Complete LLM Stack with Vision"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Start vLLM + Proxy first (existing setup)
echo "Starting coding model stack..."
./start-all-vllm.sh $MODEL_ARG $DEBUG_FLAG &
VLLM_STACK_PID=$!

# Wait for services to be ready
echo ""
echo "Waiting for coding stack to be ready..."
sleep 10

# Check if compression proxy is up
for i in {1..30}; do
    if lsof -i:8002 >/dev/null 2>&1; then
        echo "✓ Compression proxy is ready!"
        break
    fi
    sleep 1
done

echo ""

# Start vision API
echo "Starting vision API server..."
./start-vision-api.sh $DEBUG_FLAG &
VISION_PID=$!

# Wait for vision API
echo ""
echo "Waiting for vision API to start (checking port 8004)..."
for i in {1..30}; do
    if lsof -i:8004 >/dev/null 2>&1; then
        echo "✓ Vision API is ready!"
        break
    fi
    sleep 1
    echo -n "."
done

echo ""
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Complete LLM Stack Ready!"
echo "════════════════════════════════════════════════════════════════"
echo "  vLLM Server:        http://localhost:8000"
echo "  Compression Proxy:  http://localhost:8002"
echo "  Vision API:         http://localhost:8004"
echo ""
echo "  📝 Cursor IDE Configuration:"
echo "    API Endpoint: http://localhost:8002/v1"
echo "    Model Name:   $(cat "$ROOT/.llm-config" | grep "^key=" | head -1 | cut -d= -f2)"
echo ""
echo "  ✨ Features:"
echo "    • Text completions (GPU, fast)"
echo "    • Tool calling (integrated)"
echo "    • Vision/Image analysis (CPU, slower)"
echo ""
echo "  Logs:"
echo "    tail -f $ROOT/vllm-server.log"
echo "    tail -f $ROOT/compression-proxy.log"
echo "    tail -f $ROOT/vision-api.log"
echo ""
echo "  To stop:"
echo "    ./stop-all.sh"
echo "════════════════════════════════════════════════════════════════"
