#!/bin/bash
# Start vLLM with extended context mode (128K with CPU offloading)
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

echo "════════════════════════════════════════════════════════════════"
echo "  ⚠️  EXTENDED CONTEXT MODE (128K with CPU offloading)"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This mode enables larger context windows using:"
echo "  • YaRN RoPE scaling (4x context extension)"
echo "  • CPU KV cache offloading"
echo ""
echo "⚠️  WARNING: Performance will be 50-70% slower!"
echo "   - Token generation: ~20-30 tok/s (vs 40-60 normal)"
echo "   - Tool calls: 3-5x slower"
echo "   - Noticeable typing lag"
echo ""
echo "💡 For 50K+ token contexts, consider:"
echo "   - Tool-based context (model reads files as needed)"
echo "   - RAG with semantic search"
echo ""
read -p "Continue with extended mode? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Cancelled. Use './start-all-vllm.sh' for normal mode."
    exit 0
fi

# Export extended mode flag
export EXTENDED_CONTEXT_MODE="1"

# Call normal startup which will detect this flag
exec "$ROOT/start-all-vllm.sh" "$@"
