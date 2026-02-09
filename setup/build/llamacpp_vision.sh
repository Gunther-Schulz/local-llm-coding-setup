#!/usr/bin/env bash
# Build llama.cpp (CPU) for vision API. Invoked by setup/install.sh; not meant to be run directly.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LLAMACPP_DIR="$ROOT/external/llama.cpp"
BUILD_DIR="$LLAMACPP_DIR/build"

echo "════════════════════════════════════════════════════════════════"
echo "  Building llama.cpp for Vision Support"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Clone llama.cpp if needed
if [ ! -d "$LLAMACPP_DIR" ]; then
    echo "Cloning llama.cpp..."
    mkdir -p external
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMACPP_DIR"
    echo "✓ Cloned llama.cpp"
    echo ""
else
    echo "✓ llama.cpp already at $LLAMACPP_DIR"
    echo "  Pulling latest..."
    (cd "$LLAMACPP_DIR" && git fetch origin && git checkout master 2>/dev/null; git pull origin master 2>/dev/null) || true
    echo ""
fi

# Check if already built (skip unless FORCE_LLAMACPP_REBUILD=1)
if [ -z "$FORCE_LLAMACPP_REBUILD" ]; then
    if [ -f "$BUILD_DIR/bin/llama-mtmd-cli" ] || [ -f "$BUILD_DIR/bin/llama-llava-cli" ]; then
        BINARY=$([ -f "$BUILD_DIR/bin/llama-mtmd-cli" ] && echo "llama-mtmd-cli" || echo "llama-llava-cli")
        echo "✓ llama.cpp already built at: $BUILD_DIR/bin/$BINARY"
        echo "  (To rebuild: FORCE_LLAMACPP_REBUILD=1 or ./setup/build/update_llamacpp.sh)"
        echo ""
        exit 0
    fi
fi

echo "Building llama.cpp (CPU, vision)..."
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DBUILD_SHARED_LIBS=OFF

cmake --build . --config Release -j$(nproc)

echo ""
echo "✓ llama.cpp (vision) built: $BUILD_DIR/bin/"
echo ""
