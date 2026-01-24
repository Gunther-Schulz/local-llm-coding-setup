#!/bin/bash
# Toggle context mode without changing model

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
source "$ROOT/lib/config-manager.sh"

# Get current mode
CURRENT_MODE=$(get_extended_context_mode)

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Toggle Context Mode"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [[ "$CURRENT_MODE" == "1" ]]; then
    echo "Current: 🟡 Extended Mode (128K, slower)"
    echo ""
    read -p "Switch to Normal Mode (32K, faster)? [Y/n]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] || [[ -z $REPLY ]]; then
        set_context_mode "normal"
        echo ""
        echo "✓ Switched to Normal Mode"
        echo "  Restart servers for change to take effect:"
        echo "  ./stop-vllm.sh && ./start-all-vllm.sh"
    fi
else
    echo "Current: 🟢 Normal Mode (32K, fast)"
    echo ""
    echo "⚠️  Extended mode is 50-70% slower!"
    echo ""
    read -p "Switch to Extended Mode (128K, slower)? [y/N]: " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        set_context_mode "extended"
        echo ""
        echo "✓ Switched to Extended Mode"
        echo "  Restart servers for change to take effect:"
        echo "  ./stop-vllm.sh && ./start-all-vllm.sh"
    fi
fi

echo ""
