#!/usr/bin/env bash
# Build llama-server from the GLM 4.5 tool-call fix branch (parse-only for AUTO).
# Use this to get the fix for "Grammar still awaiting trigger" / llama.cpp #19068.
#
# Usage: ./setup/build/rebuild_llamacpp_glm_fix.sh
#
# - Checks out fix/glm45-tool-parse-only-auto in external/llama.cpp (creates from origin if needed).
# - Forces a full CUDA rebuild.
# Conda env is unchanged (llama.cpp is built from source, not pip).
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LLAMACPP_DIR="$ROOT/external/llama.cpp"
BRANCH="${LLAMACPP_GLM_FIX_BRANCH:-fix/glm45-tool-parse-only-auto}"

echo "════════════════════════════════════════════════════════════════"
echo "  Build llama.cpp from branch: $BRANCH (GLM 4.5 tool-call fix)"
echo "════════════════════════════════════════════════════════════════"
echo ""

if [ ! -d "$LLAMACPP_DIR" ]; then
  echo "llama.cpp not found at $LLAMACPP_DIR"
  echo "Run full setup first: ./setup/install.sh"
  exit 1
fi

echo "Checking out branch: $BRANCH"
(cd "$LLAMACPP_DIR" && git fetch origin 2>/dev/null; git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH" 2>/dev/null) || {
  echo "Branch $BRANCH not found locally or on origin."
  echo "Create it in external/llama.cpp (e.g. git checkout -b $BRANCH) and re-run."
  exit 1
}
echo "✓ Branch $BRANCH checked out"
echo ""

export LLAMACPP_BRANCH="$BRANCH"
export FORCE_LLAMACPP_REBUILD=1
"$ROOT/setup/build/llamacpp_cuda.sh"

echo ""
echo "✓ llama-server built from $BRANCH: $LLAMACPP_DIR/build-cuda/bin/llama-server"
