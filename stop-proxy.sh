#!/bin/bash
# Stop only the compression proxy

echo "Stopping compression proxy..."
pkill -9 -f compression_proxy && echo "✓ Compression proxy stopped" || echo "✗ No compression proxy running"

# Check if port 8002 is free
if lsof -ti:8002 >/dev/null 2>&1; then
    echo "⚠ Port 8002 still in use:"
    lsof -i:8002
else
    echo "✓ Port 8002 is free"
fi
