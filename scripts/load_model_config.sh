#!/usr/bin/env bash
# Load config/models/<model_key>.yaml and print shell export statements for run_server.sh.
# Usage: eval "$(scripts/load_model_config.sh <model_key>)"
# Requires: python3, PyYAML (pip install pyyaml).
# Requires: nested YAML with backend + llama: {...} and vllm: {...} dicts (no flat fallback).

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

def emit(env_var, val):
    if val is None:
        return
    if isinstance(val, bool):
        val = "1" if val else "0"
    if isinstance(val, list):
        val = " ".join(str(x) for x in val)
    print(f"export {env_var}={shell_quote(val)}")

path = sys.argv[1]
with open(path) as f:
    data = yaml.safe_load(f) or {}

backend = data.get("backend", "llama")
# Nested only: llama: {...} and vllm: {...} dicts.
llama_block = data.get("llama") if isinstance(data.get("llama"), dict) else {}
vllm_block = data.get("vllm") if isinstance(data.get("vllm"), dict) else {}

emit("BACKEND", backend)

# Shared (top-level): apply to both backends
for k, env in [("temp", "TEMP"), ("top_p", "TOP_P"), ("top_k", "TOP_K"), ("min_p", "MIN_P")]:
    v = data.get(k)
    if v is not None:
        emit(env, v)

if backend == "llama":
    # Keys under llama: (yaml_key, ENV_VAR)
    for yk, ev in [
        ("gguf", "GGUF"),
        ("mmproj", "MMPROJ"),
        ("context_size", "CONTEXT_SIZE"),
        ("n_gpu_layers", "N_GPU_LAYERS"),
        ("threads", "THREADS"),
        ("jinja", "JINJA"),
        ("repeat_penalty", "REPEAT_PENALTY"),
        ("seed", "SEED"),
        ("batch_size", "BATCH_SIZE"),
        ("ubatch_size", "UBATCH_SIZE"),
        ("chat_template_file", "CHAT_TEMPLATE_FILE"),
        ("flash_attn", "FLASH_ATTN"),
        ("cache_type_k", "CACHE_TYPE_K"),
        ("cache_type_v", "CACHE_TYPE_V"),
    ]:
        v = llama_block.get(yk)
        if v is not None:
            emit(ev, v)
else:
    # backend == vllm: keys under vllm: (yaml_key, ENV_VAR)
    for yk, ev in [
        ("model", "VLLM_MODEL"),
        ("tool_call_parser", "VLLM_TOOL_CALL_PARSER"),
        ("max_model_len", "VLLM_MAX_MODEL_LEN"),
        ("tensor_parallel", "VLLM_TENSOR_PARALLEL"),
        ("gpu_memory_utilization", "VLLM_GPU_MEMORY_UTILIZATION"),
        ("serve_extra", "VLLM_SERVE_EXTRA"),
        ("tokenizer", "VLLM_TOKENIZER"),
    ]:
        v = vllm_block.get(yk)
        if v is not None:
            emit(ev, v)
PY
