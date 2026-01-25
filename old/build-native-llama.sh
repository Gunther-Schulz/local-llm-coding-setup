#!/bin/bash
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

echo "Building native llama.cpp with CUDA support..."

cd "$ROOT"

# Clone llama.cpp if not exists (official upstream)
if [ ! -d "llama.cpp-native" ]; then
    echo "Cloning official llama.cpp (ggerganov/llama.cpp)..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git llama.cpp-native
fi

cd llama.cpp-native

# Check if already built
if [ -f "build/bin/llama-server" ]; then
    echo "✅ llama-server already built!"
    echo "Location: $ROOT/llama.cpp-native/build/bin/llama-server"
    exit 0
fi

echo "Detecting GPU compute capability..."
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "GPU Compute Capability: ${COMPUTE_CAP}"

echo "Installing ccache for faster compilation (optional)..."
sudo pacman -S --needed --noconfirm ccache 2>/dev/null || true

echo "Building with CUDA support using CMake (this will take 5-10 minutes)..."

# Clean previous build
rm -rf build

# CMake configure with CUDA
cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES=${COMPUTE_CAP/./} \
    -DCMAKE_BUILD_TYPE=Release

# Build llama-server
cmake --build build --config Release --target llama-server -j$(nproc)

echo ""
echo "✅ Build complete!"
echo ""
ls -lh build/bin/llama-server
echo ""
echo "Test it with:"
echo "./build/bin/llama-server --help"

