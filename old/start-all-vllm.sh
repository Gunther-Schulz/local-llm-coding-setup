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

# Load config manager and model selector library
source "$ROOT/lib/config-manager.sh"
source "$ROOT/lib/model-selector.sh"

# Determine which model to use
SELECTED_MODEL=""
if [[ -z "$MODEL_KEY" ]]; then
    # No -m flag: use saved config
    SELECTED_MODEL=$(get_current_model)
    
    if [[ -z "$SELECTED_MODEL" ]]; then
        echo "⚠️  No model selected yet!"
        echo ""
        echo "Please run: ./select-model.sh"
        echo "Or use: $0 -m MODEL_KEY"
        echo ""
        exit 1
    fi
    
    echo "Using saved model: $SELECTED_MODEL"
    echo "(Change with: ./select-model.sh)"
    echo ""
else
    # -m flag provided: use override
    SELECTED_MODEL="$MODEL_KEY"
    echo "Using override model: $SELECTED_MODEL"
    echo "(Saved model will be used on next normal start)"
    echo ""
fi

# Export model configuration for both services
export_model_config "$SELECTED_MODEL" >/dev/null

# Load context mode from centralized config
export EXTENDED_CONTEXT_MODE=$(get_extended_context_mode)

# If extended mode, update MODEL_MAX_CONTEXT for proxy
if [[ "$EXTENDED_CONTEXT_MODE" == "1" && -n "$MODEL_EXTENDED_CONTEXT" ]]; then
    export MODEL_MAX_CONTEXT="$MODEL_EXTENDED_CONTEXT"
    echo "ℹ️  Extended context mode: Using ${MODEL_MAX_CONTEXT} tokens"
    echo ""
fi

# Start vLLM server in background (pass -m flag if used)
echo "Starting vLLM server..."
if [[ -n "$MODEL_KEY" ]]; then
    ./start-vllm-server.sh -m "$MODEL_KEY" &
else
    ./start-vllm-server.sh &
fi
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

# Start compression proxy in background
# (Model config and context mode already exported above)
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
