#!/bin/bash
# Build llama.cpp with multimodal support for vision API
# This is a one-time setup for the vision server
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
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
    echo "To rebuild, delete: $BUILD_DIR"
    echo "Then run this script again."
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

# Build
echo "Building llama.cpp (this takes ~5-10 minutes)..."
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF \
    -DBUILD_SHARED_LIBS=OFF

cmake --build . --config Release -j$(nproc)

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ llama.cpp Built Successfully!"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Check which binary was created
if [ -f "$BUILD_DIR/bin/llama-mtmd-cli" ]; then
    BINARY="llama-mtmd-cli"
elif [ -f "$BUILD_DIR/bin/llama-llava-cli" ]; then
    BINARY="llama-llava-cli"
else
    echo "⚠️  Warning: No multimodal binary found in $BUILD_DIR/bin/"
    BINARY="(not found)"
fi

echo "Binary location: $BUILD_DIR/bin/$BINARY"
echo ""
echo "This binary is used by the vision server (./run/vision.sh)"
echo "to process images on CPU."
echo ""
echo "Next steps:"
echo "  1. Download a vision model: ./stack/download_vision_model.sh qwen2-vl-2b-q4"
echo "  2. Select it: ./run/select_vision_model.sh"
echo "  3. Start vision server: ./run/vision.sh"
echo "════════════════════════════════════════════════════════════════"
