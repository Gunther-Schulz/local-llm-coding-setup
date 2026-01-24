#!/bin/bash
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"

# Start an OpenAI-compatible vLLM server using the existing GGUF model.
# This uses vLLM's experimental GGUF support so we can reuse the local
# Qwen2.5-Coder-14B Q4_K_M GGUF without downloading a new checkpoint.
# You can override via:
#   export VLLM_GGUF_MODEL="/path/to/model.gguf"
#   export VLLM_TOKENIZER_ID="Qwen/Qwen2.5-Coder-14B-Instruct"
#   # NOTE: With GGUF, vLLM enforces the model's native training context (32K for Qwen2.5-Coder-14B).
#   # Setting VLLM_MAX_LEN above 32768 will cause a ValidationError.
#   export VLLM_MAX_LEN=32768
#   export VLLM_DTYPE=float16   # GGUF supports float16/float32 only; auto picks bfloat16 on Blackwell and fails.

GGUF_MODEL="${VLLM_GGUF_MODEL:-$ROOT/models/qwen2.5-coder-14b-q4_k_m/qwen2.5-coder-14b-instruct-q4_k_m.gguf}"
DTYPE="${VLLM_DTYPE:-float16}"
TOKENIZER_ID="${VLLM_TOKENIZER_ID:-Qwen/Qwen2.5-Coder-14B-Instruct}"
# vLLM with GGUF is limited to native training context (no RoPE scaling support)
MAX_LEN="${VLLM_MAX_LEN:-32768}"
HOST="${VLLM_HOST:-0.0.0.0}"
PORT="${VLLM_PORT:-8000}"

cd "$ROOT"

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
  : # already in a venv
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/venv/bin/activate"
else
  echo "ERROR: No conda (env llm) or venv found. Create one:"
  echo "  conda create -n llm python=3.10 -y && conda activate llm"
  echo "  or: python3 -m venv $ROOT/.venv && . $ROOT/.venv/bin/activate"
  echo "Then: ./install-deps.sh   and   ./setup-vllm.sh"
  exit 1
fi

echo "🚀 Starting vLLM OpenAI server..."
echo "  GGUF model     : $GGUF_MODEL"
echo "  Served as      : qwen2.5-coder-14b"
echo "  Tokenizer      : $TOKENIZER_ID"
echo "  dtype          : $DTYPE (float16 required for GGUF on Blackwell/5090)"
echo "  Max model len  : $MAX_LEN (native context - GGUF doesn't support RoPE scaling)"
echo "  Host / Port    : $HOST:$PORT"
echo ""

python -m vllm.entrypoints.openai.api_server \
  --model "$GGUF_MODEL" \
  --tokenizer "$TOKENIZER_ID" \
  --served-model-name "qwen2.5-coder-14b" \
  --host "$HOST" \
  --port "$PORT" \
  --dtype "$DTYPE" \
  --max-model-len "$MAX_LEN" \
  --tensor-parallel-size 1 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder


