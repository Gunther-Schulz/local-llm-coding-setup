#!/bin/bash
# Build llama.cpp with CUDA for GPU inference (benchmark subdir)
set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/.." && pwd)"
LLAMACPP_DIR="$BENCH_DIR/llama.cpp"
BUILD_DIR="$LLAMACPP_DIR/build"

# Prefer CUDA_HOME from env (e.g. /opt/cuda); else find nvcc
if [ -n "$CUDA_HOME" ]; then
  NVCC="$CUDA_HOME/bin/nvcc"
else
  for d in /opt/cuda /usr/local/cuda; do
    if [ -x "$d/bin/nvcc" ]; then
      export CUDA_HOME="$d"
      NVCC="$d/bin/nvcc"
      break
    fi
  done
fi
if [ -z "$NVCC" ] || [ ! -x "$NVCC" ]; then
  echo "No nvcc found. Set CUDA_HOME or install CUDA toolkit."
  exit 1
fi

echo "Using CUDA_HOME=$CUDA_HOME"
echo "Building llama.cpp with CUDA in $LLAMACPP_DIR"
echo ""

if [ ! -d "$LLAMACPP_DIR" ]; then
  echo "Cloning llama.cpp..."
  git clone --depth 1 https://github.com/ggerganov/llama.cpp.git "$LLAMACPP_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

# GGML_CUDA=ON for modern llama.cpp; fallback LLAMA_CUBLAS for older
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DBUILD_SHARED_LIBS=OFF

cmake --build . --config Release -j$(nproc)

echo ""
echo "Build done. Binary: $BUILD_DIR/bin/llama-cli"
