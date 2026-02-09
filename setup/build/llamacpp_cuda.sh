#!/usr/bin/env bash
# Build llama.cpp (CUDA) for LLM engine and benchmarks. Invoked by setup/install.sh; not meant to be run directly.
set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

LLAMACPP_DIR="$ROOT/external/llama.cpp"
BUILD_DIR="$LLAMACPP_DIR/build-cuda"

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
  echo "No nvcc found. Set CUDA_HOME or install CUDA toolkit. Skipping llama.cpp CUDA build."
  exit 0
fi

echo "════════════════════════════════════════════════════════════════"
echo "  Building llama.cpp (CUDA) for LLM engine & benchmarks"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Using CUDA_HOME=$CUDA_HOME"
echo ""

# Clone llama.cpp if needed (shared with vision build)
if [ ! -d "$LLAMACPP_DIR" ]; then
  echo "Cloning llama.cpp..."
  mkdir -p external
  git clone https://github.com/ggerganov/llama.cpp.git "$LLAMACPP_DIR"
  echo "✓ Cloned llama.cpp"
  echo ""
else
  echo "✓ llama.cpp already at $LLAMACPP_DIR"
  echo "  Pulling latest (for native tool-call parsing, e.g. Qwen3 Coder)..."
  (cd "$LLAMACPP_DIR" && git fetch origin && git checkout master 2>/dev/null; git pull origin master 2>/dev/null) || true
  echo ""
fi

# Check if already built (skip unless FORCE_LLAMACPP_REBUILD=1)
if [ -z "$FORCE_LLAMACPP_REBUILD" ] && [ -x "$BUILD_DIR/bin/llama-server" ] && [ -x "$BUILD_DIR/bin/llama-bench" ]; then
  echo "✓ CUDA build already present: $BUILD_DIR/bin/"
  echo "  (To get native tool-call parsing for Qwen: rm -rf $BUILD_DIR and run this script again, or FORCE_LLAMACPP_REBUILD=1)"
  echo ""
  exit 0
fi

echo "Building llama.cpp with CUDA..."
echo ""

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$CUDA_HOME/bin/nvcc" \
  -DBUILD_SHARED_LIBS=OFF

cmake --build . --config Release -j$(nproc)

echo ""
echo "✓ llama.cpp (CUDA) built: $BUILD_DIR/bin/"
echo ""
