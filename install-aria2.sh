#!/bin/bash
# Install aria2c for ultra-fast model downloads

echo "════════════════════════════════════════════════════════════════"
echo "  🚀 Installing aria2c (Multi-Connection Downloader)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# Detect OS and install
if command -v pacman &> /dev/null; then
    echo "Detected CachyOS/Arch Linux..."
    sudo pacman -S --needed aria2
elif command -v apt &> /dev/null; then
    echo "Detected Ubuntu/Debian..."
    sudo apt update && sudo apt install -y aria2
elif command -v dnf &> /dev/null; then
    echo "Detected Fedora/RHEL..."
    sudo dnf install -y aria2
elif command -v brew &> /dev/null; then
    echo "Detected macOS..."
    brew install aria2
else
    echo "❌ Unsupported OS. Please install aria2 manually:"
    echo "  https://aria2.github.io/"
    exit 1
fi

# Verify installation
if command -v aria2c &> /dev/null; then
    echo ""
    echo "════════════════════════════════════════════════════════════════"
    echo "  ✅ aria2c Installed Successfully!"
    echo "════════════════════════════════════════════════════════════════"
    echo ""
    echo "Version: $(aria2c --version | head -1)"
    echo ""
    echo "Benefits:"
    echo "  • 16 parallel connections per download"
    echo "  • 10-50x faster than wget/curl"
    echo "  • Resume support"
    echo "  • Automatic chunk splitting"
    echo ""
    echo "Expected speeds:"
    echo "  wget/curl:  10-50 MB/s"
    echo "  aria2c:     100-500 MB/s (10x faster!)"
    echo ""
    echo "Now run: ./download-model.sh"
    echo "════════════════════════════════════════════════════════════════"
else
    echo "❌ Installation failed"
    exit 1
fi
