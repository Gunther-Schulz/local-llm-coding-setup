#!/bin/bash
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

# Check if already built (check for new or old binary name)
if [ -f "$BUILD_DIR/bin/llama-mtmd-cli" ] || [ -f "$BUILD_DIR/bin/llama-llava-cli" ]; then
    BINARY=$([ -f "$BUILD_DIR/bin/llama-mtmd-cli" ] && echo "llama-mtmd-cli" || echo "llama-llava-cli")
    echo "✓ llama.cpp already built at: $BUILD_DIR/bin/$BINARY"
    echo ""
    exit 0
fi

# Clone llama.cpp if needed
if [ ! -d "$LLAMACPP_DIR" ]; then
    echo "Cloning llama.cpp..."
    mkdir -p external
    git clone https://github.com/ggerganov/llama.cpp.git "$LLAMACPP_DIR"
    echo "✓ Cloned llama.cpp"
    echo ""
else
    echo "✓ llama.cpp already cloned"
    echo ""
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
