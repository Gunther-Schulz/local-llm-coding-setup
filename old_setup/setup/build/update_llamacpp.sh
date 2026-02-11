#!/usr/bin/env bash
# Update llama.cpp to latest master and rebuild CUDA (llama-server).
# Usage: ./setup/build/update_llamacpp.sh
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LLAMACPP_DIR="$ROOT/external/llama.cpp"

echo "════════════════════════════════════════════════════════════════"
echo "  Update llama.cpp to latest and rebuild (CUDA)"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ ! -d "$LLAMACPP_DIR" ]; then
  echo "llama.cpp not found at $LLAMACPP_DIR"
  echo "Run full setup first: ./setup/install.sh"
  exit 1
fi

echo "Pulling latest from origin master..."
(cd "$LLAMACPP_DIR" && git fetch origin && git checkout master 2>/dev/null; git pull origin master)
echo "✓ Updated llama.cpp"
echo ""

export FORCE_LLAMACPP_REBUILD=1
echo "Rebuilding llama.cpp (CUDA)..."
"$ROOT/setup/build/llamacpp_cuda.sh"
echo ""
echo "✓ llama.cpp update complete."
