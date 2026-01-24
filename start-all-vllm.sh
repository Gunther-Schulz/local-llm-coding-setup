#!/bin/bash
# Start both vLLM server and compression proxy

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

# Parse arguments for model selection and debug mode
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
echo "  🚀 Starting Complete LLM Stack"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check if model is selected
CURRENT_MODEL_FILE="$ROOT/.current-model"
if [[ -z "$MODEL_ARG" && ! -f "$CURRENT_MODEL_FILE" ]]; then
    echo "⚠️  No model selected yet!"
    echo ""
    echo "Please run: ./select-model.sh"
    echo "Or use: $0 -m MODEL_KEY"
    echo ""
    exit 1
fi

if [[ -z "$MODEL_ARG" ]]; then
    SELECTED_MODEL=$(cat "$CURRENT_MODEL_FILE")
    echo "Using model: $SELECTED_MODEL"
    echo "(Change with: ./select-model.sh or use -m flag)"
    echo ""
fi

# Start vLLM server in background
echo "Starting vLLM server..."
./start-vllm-server.sh $MODEL_ARG &
VLLM_PID=$!

# Wait for vLLM to be ready
echo ""
echo "Waiting for vLLM server to start (checking port 8000)..."
for i in {1..60}; do
    if lsof -i:8000 >/dev/null 2>&1; then
        echo "✓ vLLM server is ready!"
        break
    fi
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "✗ vLLM server process died. Check logs:"
        echo "  tail -100 $ROOT/vllm-server.log"
        exit 1
    fi
    sleep 2
    echo -n "."
done

if ! lsof -i:8000 >/dev/null 2>&1; then
    echo ""
    echo "✗ vLLM server failed to start after 120 seconds"
    exit 1
fi

echo ""
echo ""

# Export model config for compression proxy
# If -m flag was used, we need to export the config so proxy uses the same model
if [[ -n "$MODEL_KEY" ]]; then
    source "$ROOT/lib/model-selector.sh"
    export_model_config "$MODEL_KEY" >/dev/null
fi

# Start compression proxy in background
echo "Starting compression proxy..."
./start-compression-proxy.sh $DEBUG_FLAG &

# Wait for proxy to be ready
echo ""
echo "Waiting for compression proxy to start (checking port 8002)..."
for i in {1..30}; do
    if lsof -i:8002 >/dev/null 2>&1; then
        echo "✓ Compression proxy is ready!"
        break
    fi
    sleep 1
    echo -n "."
done

echo ""
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ LLM Stack Ready!"
echo "════════════════════════════════════════════════════════════════"
echo "  vLLM Server:        http://localhost:8000"
echo "  Compression Proxy:  http://localhost:8002"
echo ""
echo "  📝 Cursor IDE Configuration:"
echo "    API Endpoint: http://localhost:8002/v1"
echo "    Model Name:   $SELECTED_MODEL"
echo ""
echo "  Logs:"
echo "    tail -f $ROOT/vllm-server.log"
echo "    tail -f $ROOT/compression-proxy.log"
echo ""
echo "  To stop:"
echo "    ./stop-all.sh"
echo "════════════════════════════════════════════════════════════════"
