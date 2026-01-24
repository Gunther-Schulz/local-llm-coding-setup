#!/bin/bash
set -e

ROOT="${WORKSPACE:-${RUNPOD_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)}}"
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
  :
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  # shellcheck disable=SC1090
  . "$ROOT/venv/bin/activate"
else
  echo "ERROR: No conda or venv. Create one first:"
  echo "  conda create -n llm python=3.10 -y && conda activate llm"
  echo "  or: python3 -m venv $ROOT/.venv && . $ROOT/.venv/bin/activate"
  exit 1
fi

echo "📦 Installing Python dependencies from requirements.txt..."

if [ ! -f "$ROOT/requirements.txt" ]; then
  echo "ERROR: $ROOT/requirements.txt not found."
  exit 1
fi

python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt

echo ""
echo "✅ Dependency installation complete."


