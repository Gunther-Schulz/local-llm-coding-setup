#!/bin/bash
# Install HuggingFace CLI tools for faster downloads

set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

# Activate conda environment
_conda_sh=""
for d in "$ROOT/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.miniconda3" "/workspace/miniconda3"; do
  if [ -f "${d}/etc/profile.d/conda.sh" ]; then
    _conda_sh="${d}/etc/profile.d/conda.sh"
    break
  fi
done
[ -z "$_conda_sh" ] && command -v conda &>/dev/null && {
  _base=$(conda info --base 2>/dev/null)
  [ -n "$_base" ] && [ -f "${_base}/etc/profile.d/conda.sh" ] && _conda_sh="${_base}/etc/profile.d/conda.sh"
}

if [ -n "$_conda_sh" ]; then
  source "$_conda_sh"
  conda activate llm
else
  echo "ERROR: Conda environment not found"
  exit 1
fi

echo "════════════════════════════════════════════════════════════════"
echo "  📥 Installing HuggingFace CLI + hf_transfer"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "This will enable MUCH faster model downloads (multi-threaded)!"
echo ""

# Install HuggingFace CLI with hf_transfer support
pip install -U "huggingface_hub[cli]" hf-transfer

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✅ Installation Complete!"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "Benefits:"
echo "  • Multi-threaded downloads (much faster)"
echo "  • Resume support"
echo "  • Better error handling"
echo "  • Direct HuggingFace integration"
echo ""
echo "Now run: ./download-model.sh"
echo "════════════════════════════════════════════════════════════════"
