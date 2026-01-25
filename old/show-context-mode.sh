#!/bin/bash
# Show current context mode setting

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
source "$ROOT/lib/config-manager.sh"

MODE=$(get_context_mode)

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Current Context Mode Setting"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [[ "$MODE" == "extended" ]]; then
    echo "  🟡 EXTENDED MODE (128K context)"
    echo "     • Slower performance (20-30 tok/s)"
    echo "     • GPU + CPU offloading"
    echo "     • For large contexts (50K+ tokens)"
else
    echo "  🟢 NORMAL MODE (32K context)"
    echo "     • Fast performance (40-60 tok/s)"
    echo "     • GPU only"
    echo "     • Recommended for most tasks"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "To change: ./select-model.sh"
echo ""
