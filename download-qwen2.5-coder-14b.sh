#!/bin/bash
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

# Prefer aria2c (multi-connection) or curl to avoid hf download stalls ("first chunk then stall").
# For gated models: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN.
# Install aria2 for best results: sudo pacman -S aria2

MODEL_REPO="yemiao2745/Qwen2.5-Coder-14B-Instruct-Q4_K_M-GGUF"
MODEL_FILE="qwen2.5-coder-14b-instruct-q4_k_m.gguf"
TARGET_DIR="$ROOT/models/qwen2.5-coder-14b-q4_k_m"
URL="https://huggingface.co/${MODEL_REPO}/resolve/main/${MODEL_FILE}"
TOKEN="${HF_TOKEN:-$HUGGING_FACE_HUB_TOKEN}"

echo "📥 Downloading Qwen2.5-Coder-14B-Instruct Q4_K_M GGUF..."
echo "  Repo : $MODEL_REPO"
echo "  File : $MODEL_FILE"
echo "  Dest : $TARGET_DIR"
echo ""

mkdir -p "$TARGET_DIR"

if command -v aria2c &>/dev/null; then
  echo "Using aria2c (multi-connection, resume)."
  ARIA2_EXTRA=()
  [[ -n "$TOKEN" ]] && ARIA2_EXTRA=(-H "Authorization: Bearer $TOKEN")
  if aria2c -x 16 -s 16 -c --allow-overwrite=true -d "$TARGET_DIR" -o "$MODEL_FILE" "${ARIA2_EXTRA[@]}" "$URL"; then
    echo ""
    echo "✅ Download complete (aria2c)."
    echo "Model at: $TARGET_DIR/$MODEL_FILE"
    exit 0
  fi
  echo "aria2c failed, trying fallback..."
fi

if command -v curl &>/dev/null; then
  echo "Using curl (resume)."
  CURL_EXTRA=()
  [[ -n "$TOKEN" ]] && CURL_EXTRA=(-H "Authorization: Bearer $TOKEN")
  if curl -fL -C - -o "$TARGET_DIR/$MODEL_FILE" "${CURL_EXTRA[@]}" "$URL"; then
    echo ""
    echo "✅ Download complete (curl)."
    echo "Model at: $TARGET_DIR/$MODEL_FILE"
    exit 0
  fi
  echo "curl failed, trying hf download..."
fi

echo "Using hf download (fallback; can stall on some networks)."
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_HUB_ENABLE_EMERGENCY_RETRY=1

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
  # shellcheck disable=SC1090
  . "$_conda_sh"
  conda activate llm
elif [ -n "$VIRTUAL_ENV" ]; then
  :
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/venv/bin/activate"
fi

hf download "$MODEL_REPO" --include "$MODEL_FILE" --local-dir "$TARGET_DIR"

echo ""
echo "✅ Download complete."
echo "Model at: $TARGET_DIR/$MODEL_FILE"
