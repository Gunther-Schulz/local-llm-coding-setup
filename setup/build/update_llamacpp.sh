#!/bin/bash
# Update llama.cpp to latest master and rebuild (vision + CUDA).
# Usage: ./setup/build/update_llamacpp.sh
# Optional: LLAMACPP_UPDATE_VISION=0 or LLAMACPP_UPDATE_CUDA=0 to skip one build.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LLAMACPP_DIR="$ROOT/external/llama.cpp"

echo "════════════════════════════════════════════════════════════════"
echo "  Update llama.cpp to latest and rebuild"
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

if [ "${LLAMACPP_UPDATE_VISION:-1}" = "1" ]; then
  echo "Rebuilding llama.cpp (vision)..."
  "$ROOT/setup/build/llamacpp_vision.sh"
  echo ""
fi

if [ "${LLAMACPP_UPDATE_CUDA:-1}" = "1" ]; then
  echo "Rebuilding llama.cpp (CUDA)..."
  "$ROOT/setup/build/llamacpp_cuda.sh"
  echo ""
fi

echo "✓ llama.cpp update complete."
