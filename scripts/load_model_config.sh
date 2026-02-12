#!/usr/bin/env bash
# Load config/models/<model_key>.yaml and print shell export statements for run_server.sh.
# Usage: eval "$(scripts/load_model_config.sh <model_key>)"
# Requires: python3, PyYAML (pip install pyyaml).

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
model_key="${1:?Usage: load_model_config.sh <model_key>}"
yaml_file="$ROOT/config/models/${model_key}.yaml"
if [[ ! -f "$yaml_file" ]]; then
  echo "Model config not found: $yaml_file" >&2
  exit 1
fi
python3 - "$yaml_file" << 'PY'
import sys
import yaml

def shell_quote(s):
    if s is None:
        return "''"
    s = str(s)
    if not s or s.strip() != s or "'" in s or " " in s or "$" in s:
        return "'" + s.replace("'", "'\"'\"'") + "'"
    return s

path = sys.argv[1]
with open(path) as f:
    data = yaml.safe_load(f) or {}

# Map YAML keys to env vars run_server expects (UPPERCASE)
mapping = [
    ("gguf", "GGUF"),
    ("context_size", "CONTEXT_SIZE"),
    ("n_gpu_layers", "N_GPU_LAYERS"),
    ("threads", "THREADS"),
    ("jinja", "JINJA"),
    ("temp", "TEMP"),
    ("top_p", "TOP_P"),
    ("top_k", "TOP_K"),
    ("min_p", "MIN_P"),
    ("seed", "SEED"),
    ("batch_size", "BATCH_SIZE"),
    ("ubatch_size", "UBATCH_SIZE"),
]
for yk, ev in mapping:
    v = data.get(yk)
    if v is None:
        continue
    if isinstance(v, bool):
        v = "1" if v else "0"
    print(f"export {ev}={shell_quote(v)}")
PY
