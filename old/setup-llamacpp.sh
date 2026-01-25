#!/bin/bash
# Setup llama.cpp for vision model inference on CPU

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
cd "$ROOT"

echo "════════════════════════════════════════════════════════════════"
echo "  🦙 llama.cpp Setup for Vision Models"
echo "════════════════════════════════════════════════════════════════"
echo ""

LLAMACPP_DIR="$ROOT/llama.cpp"
LLAMACPP_BIN="$LLAMACPP_DIR/build/bin/llama-mtmd-cli"

# Check if already installed (llama.cpp now uses CMake; binary is build/bin/llama-mtmd-cli)
if [[ -f "$LLAMACPP_BIN" ]]; then
    echo "✓ llama.cpp already installed"
    echo "  Binary: $LLAMACPP_BIN"
    echo ""
    echo "To rebuild: rm -rf $LLAMACPP_DIR/build && ./setup-llamacpp.sh"
    echo ""
    exit 0
fi

# Install dependencies (cmake, base-devel, git) via pacman. Skip if already present.
NEED_DEPS=
command -v cmake &>/dev/null || NEED_DEPS=1
command -v make  &>/dev/null || NEED_DEPS=1
command -v g++   &>/dev/null || command -v gcc &>/dev/null || NEED_DEPS=1
command -v git   &>/dev/null || NEED_DEPS=1

if [[ -z "$NEED_DEPS" ]]; then
    echo "✓ Build tools already installed (cmake, make, gcc/g++, git)"
else
    echo "📦 Installing build dependencies (pacman)..."
    echo "  (You can answer Y/n when pacman asks.)"
    echo ""
    set +e
    sudo pacman -S --needed cmake base-devel git
    rc=$?
    set -e
    if [[ $rc -ne 0 ]]; then
        # Re-check: maybe cmake/make/g++/git are already available (or user fixed it)
        NEED_DEPS=
        command -v cmake &>/dev/null || NEED_DEPS=1
        command -v make  &>/dev/null || NEED_DEPS=1
        command -v g++   &>/dev/null || command -v gcc &>/dev/null || NEED_DEPS=1
        command -v git  &>/dev/null || NEED_DEPS=1
        if [[ -z "$NEED_DEPS" ]]; then
            echo ""
            echo "✓ Build tools are available; continuing without pacman."
            echo ""
        else
            echo ""
            echo "⚠️  Pacman failed. If you see PGP/signature errors (e.g. CachyOS cmake):"
            echo "    sudo pacman-key --refresh-keys"
            echo "    sudo pacman -Sy archlinux-keyring"
            echo "  (CachyOS: cachyos-keyring instead of archlinux-keyring if needed.)"
            echo ""
            echo "  If cmake is already installed, fix keyring and re-run; the script will skip install."
            echo ""
            exit 1
        fi
    fi
fi

# Clone if not present
if [[ ! -d "$LLAMACPP_DIR" ]]; then
    echo ""
    echo "📥 Cloning llama.cpp..."
    git clone https://github.com/ggml-org/llama.cpp "$LLAMACPP_DIR"
fi
cd "$LLAMACPP_DIR"

echo ""
echo "🔨 Building llama.cpp with CMake (llama-mtmd-cli for Qwen2-VL)..."
echo "  Using $(nproc 2>/dev/null || echo 4) CPU cores"
echo ""

# llama.cpp deprecated the Makefile; use CMake. Binaries go to build/bin/
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j$(nproc 2>/dev/null || echo 4) --target llama-mtmd-cli

if [[ ! -f "$LLAMACPP_BIN" ]]; then
    echo ""
    echo "ERROR: Build did not produce $LLAMACPP_BIN"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ llama.cpp Setup Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Binary: $LLAMACPP_BIN"
echo ""
echo "Next steps:"
echo "  1. Download a vision model: ./download-vision-model.sh qwen2-vl-2b-q4"
echo "  2. Start vision API: ./start-vision-api.sh"
echo ""
