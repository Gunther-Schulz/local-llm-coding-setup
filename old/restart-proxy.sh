#!/bin/bash
# Quick restart of compression proxy
# Usage: ./restart-proxy.sh [-d|--debug]

cd ~/dev/local/runpod
./stop-proxy.sh
sleep 2
./start-compression-proxy.sh "$@" &

echo ""
echo "Compression proxy restarted. Watch logs with:"
echo "  tail -f ~/dev/local/runpod/compression-proxy.log"
