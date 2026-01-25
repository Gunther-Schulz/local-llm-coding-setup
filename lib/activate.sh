# Activate project env: conda "vLLM" or .venv/venv. Source after ROOT is set.
# Used by run/vllm.sh and run/select_model.sh.

_conda_sh=""
for d in "$ROOT/miniconda3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/.miniconda3" "/opt/miniconda3" "/workspace/miniconda3"; do
  [ -f "${d}/etc/profile.d/conda.sh" ] && { _conda_sh="${d}/etc/profile.d/conda.sh"; break; }
done
[ -z "$_conda_sh" ] && command -v conda &>/dev/null && {
  _b=$(conda info --base 2>/dev/null)
  [ -n "$_b" ] && [ -f "${_b}/etc/profile.d/conda.sh" ] && _conda_sh="${_b}/etc/profile.d/conda.sh"
}
if [ -n "$_conda_sh" ]; then
  . "$_conda_sh"
  conda activate vLLM
elif [ -n "$VIRTUAL_ENV" ]; then
  :
elif [ -f "$ROOT/.venv/bin/activate" ]; then
  . "$ROOT/.venv/bin/activate"
elif [ -f "$ROOT/venv/bin/activate" ]; then
  . "$ROOT/venv/bin/activate"
else
  echo "ERROR: No conda (env vLLM) or venv. Run: ./setup/install.sh" >&2
  exit 1
fi
