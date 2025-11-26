#!/bin/bash
set -e

echo "Building native llama.cpp with CUDA support..."

cd /workspace

# Clone llama.cpp if not exists
if [ ! -d "llama.cpp-native" ]; then
    echo "Cloning llama.cpp (patched fork)..."
    git clone --depth 1 https://github.com/Gunther-Schulz/llama.cpp.git llama.cpp-native
fi

cd llama.cpp-native

# Check if already built
if [ -f "build/bin/llama-server" ]; then
    echo "✅ llama-server already built!"
    echo "Location: /workspace/llama.cpp-native/build/bin/llama-server"
    exit 0
fi

echo "Detecting GPU compute capability..."
COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | head -1)
echo "GPU Compute Capability: ${COMPUTE_CAP}"

echo "Installing ccache for faster compilation..."
apt-get update -qq && apt-get install -y -qq ccache > /dev/null 2>&1 || echo "ccache install failed, continuing anyway..."

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

