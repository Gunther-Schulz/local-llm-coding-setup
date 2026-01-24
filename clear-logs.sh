#!/bin/bash
# Clear log files without restarting servers
# This truncates the files in place, so running processes can keep writing

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

echo "Clearing log files..."

# Truncate (clear) log files without deleting them
> "$ROOT/vllm-server.log"
> "$ROOT/compression-proxy.log"

echo "✓ Logs cleared:"
echo "  - vllm-server.log"
echo "  - compression-proxy.log"
echo ""
echo "Servers will continue writing to these files."
