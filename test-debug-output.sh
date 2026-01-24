#!/bin/bash
# Test that debug output works

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "Testing debug output..."
echo ""
echo "Expected to see:"
echo "  - [DEBUG] Full request"
echo "  - [DEBUG] Tokens: prompt=X, max_completion=Y"
echo "  - [DEBUG] Request to vLLM"
echo "  - [DEBUG] Response"
echo ""
echo "Tailing log file in real-time. Send a test request to see debug output."
echo "Press Ctrl+C to stop."
echo ""

tail -f compression-proxy.log
