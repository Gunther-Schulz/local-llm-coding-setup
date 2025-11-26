#!/bin/bash
# Install Python dependencies for the compression proxy (and related tooling)
# Run this on the RunPod instance from /workspace.
#
# This uses the system Python environment (no conda).

set -e

cd /workspace

echo "📦 Installing Python dependencies from requirements.txt..."

if [ ! -f "requirements.txt" ]; then
  echo "ERROR: /workspace/requirements.txt not found."
  echo "Make sure you've uploaded it (via upload-to-runpod.sh) before running this script."
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "✅ Dependency installation complete."


