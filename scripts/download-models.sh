#!/usr/bin/env bash
# Download models from config/models/*.yaml using aria2 (resume on interrupt, skip existing).
# Requires: aria2c, curl, python3, PyYAML. Usage: ./scripts/download-models.sh [MODEL_KEY ...]
#   No args = all models with llama.download_url or vllm.download_repo. Optional args = only those model_key(s).
#
# Only the active backend for each config is downloaded: backend=llama -> GGUF only; backend=vllm -> vLLM safetensors only.
# GGUF (llama): llama.download_url, llama.gguf (and mmproj, download_extra).
# vLLM: vllm.download_repo — list repo files via HF API, download each with aria2.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODELS_DIR="$ROOT/config/models"
cd "$ROOT"

if [[ ! -d "$MODELS_DIR" ]]; then
  echo "Config dir not found: $MODELS_DIR" >&2
  exit 1
fi

if ! command -v aria2c &>/dev/null; then
  echo "aria2c not found. Install aria2 for resume-capable downloads." >&2
  exit 1
fi

if ! command -v curl &>/dev/null; then
  echo "curl not found." >&2
  exit 1
fi

# List model keys from YAML filenames; optionally filter by argv.
get_model_keys() {
  for f in "$MODELS_DIR"/*.yaml; do
    [[ -f "$f" ]] || continue
    local k
    k="$(basename "$f" .yaml)"
    if [[ $# -ge 1 ]]; then
      for arg in "$@"; do
        if [[ "$k" == "$arg" ]]; then echo "$k"; break; fi
      done
    else
      echo "$k"
    fi
  done
}

# Read backend, download_url/gguf from llama:, download_repo from vllm:.
# Output: backend|download_url|gguf|extra1,extra2,...|vllm_download_repo
# Skip only if neither llama.download_url nor vllm.download_repo is set.
read_model_download() {
  python3 - "$1" << 'PY'
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    data = yaml.safe_load(f) or {}
backend = (data.get("backend") or "llama").strip().lower()
llama = data.get("llama") if isinstance(data.get("llama"), dict) else {}
vllm = data.get("vllm") if isinstance(data.get("vllm"), dict) else {}
url = (llama.get("download_url") or data.get("download_url") or "").strip()
if url and url.lower() == "none":
    url = ""
vllm_repo = (vllm.get("download_repo") or data.get("vllm_download_repo") or "").strip()
if vllm_repo and vllm_repo.lower() == "none":
    vllm_repo = ""
if not url and not vllm_repo:
    sys.exit(0)
gguf = (data.get("gguf") or llama.get("gguf") or "").strip()
extras = []
if llama.get("mmproj"):
    extras.append(str(llama["mmproj"]).strip())
for f in data.get("download_extra") or []:
    if f and str(f).strip() and str(f).strip() not in extras:
        extras.append(str(f).strip())
extras_str = ",".join(extras)
print(f"{backend}|{url}|{gguf}|{extras_str}|{vllm_repo}")
PY
}

# Returns 0 if local file is complete (exists and size matches remote Content-Length).
is_complete() {
  local local_path="$1"
  local url="$2"
  [[ ! -f "$local_path" ]] && return 1
  local size
  size=$(curl -sS -I -L "$url" 2>/dev/null | awk -v IGNORECASE=1 '/^Content-Length:/ { print $2; exit }' | tr -d '\r')
  [[ -z "$size" ]] && return 1
  [[ $(stat -c%s "$local_path" 2>/dev/null || stat -f%z "$local_path" 2>/dev/null) -eq "$size" ]] && return 0
  return 1
}

# Download one file with aria2 (resume, overwrite partial). Skips if is_complete.
download_one() {
  local url="$1"
  local dir="$2"
  local outname="$3"
  local fullpath="$dir/$outname"
  mkdir -p "$dir"
  if is_complete "$fullpath" "$url"; then
    echo "  skip (complete): $outname"
    return 0
  fi
  echo "  download (resume if partial): $outname"
  aria2c --continue=true --max-connection-per-server=16 --min-split-size=1M --split=16 \
    --file-allocation=none -d "$dir" -o "$outname" --allow-overwrite=true --console-log-level=warn "$url"
}

# Turn a repo URL + path into a direct download URL (Hugging Face resolve/main).
resolve_hf_url() {
  local repo_url="$1"
  local path="$2"
  local repo
  repo=$(echo "$repo_url" | sed -n 's|.*huggingface\.co/\([^/]*/[^/]*\).*|\1|p' | sed 's|/tree/.*||')
  [[ -z "$repo" ]] && return 1
  echo "https://huggingface.co/${repo}/resolve/main/${path}"
}

# List all file paths in a Hugging Face model repo (recursive). One path per line. Stdlib-only.
list_hf_repo_files() {
  local repo_id="$1"
  python3 - "$repo_id" << 'PY'
import sys, json, urllib.request
def list_tree(repo_id, path=""):
  url = f"https://huggingface.co/api/models/{repo_id}/tree/main"
  if path:
    url += "?path=" + path
  try:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Accept": "application/json"})) as r:
      data = json.loads(r.read().decode())
  except Exception:
    return
  for item in data:
    if not isinstance(item, dict):
      continue
    name = item.get("path") or ""
    rel = f"{path}/{name}" if path else name
    if item.get("type") == "file":
      print(rel)
    elif item.get("type") == "dir":
      list_tree(repo_id, rel)
list_tree(sys.argv[1])
PY
}

while read -r model_key; do
  [[ -z "$model_key" ]] && continue
  yaml_file="$MODELS_DIR/${model_key}.yaml"
  if [[ ! -f "$yaml_file" ]]; then
    continue
  fi
  line=$(read_model_download "$yaml_file") || true
  [[ -z "$line" ]] && echo "Model: $model_key (no download_url or vllm_download_repo), skip" && echo "" && continue
  IFS='|' read -r backend download_url gguf extras vllm_download_repo <<< "$line"
  dest_dir="$ROOT/models/${model_key}"
  echo "Model: $model_key (backend=$backend) -> $dest_dir"

  # ---- GGUF (and extras) from download_url — only when backend is llama ----
  if [[ "$backend" == "llama" ]] && [[ -n "$download_url" ]] && [[ "$download_url" == *"/resolve/main/"* && "$download_url" == *.gguf ]]; then
    # Direct .gguf link: download that file only.
    outname=$(basename "$download_url")
    download_one "$download_url" "$dest_dir" "$outname"
    expected="$gguf"
    if [[ -n "$expected" && "$outname" != "$expected" && -f "$dest_dir/$outname" && ! -f "$dest_dir/$expected" ]]; then
      ln -sf "$outname" "$dest_dir/$expected" 2>/dev/null || true
    fi
  elif [[ -n "$download_url" ]]; then
    # Repo URL: download the file(s) for this model. If gguf is multi-shard (e.g. -00001-of-00003.gguf), download all shards.
    if [[ -z "$gguf" ]]; then
      echo "  skip: repo URL but no 'gguf' filename in config (set under llama: or top-level), cannot choose a file"
    else
      if [[ "$gguf" =~ ^(.+)-([0-9]+)-of-([0-9]+)\.gguf$ ]]; then
        # Multi-shard: base name, shard index, total (zero-padded). Download shards 1..total.
        base="${BASH_REMATCH[1]}"
        total_pad="${BASH_REMATCH[3]}"
        total_num=$((10#$total_pad))
        width=${#total_pad}
        i=1
        while [[ $i -le $total_num ]]; do
          shard_name="${base}-$(printf "%0${width}d" "$i")-of-${total_pad}.gguf"
          url=$(resolve_hf_url "$download_url" "$shard_name")
          if [[ -n "$url" ]]; then
            download_one "$url" "$dest_dir" "$shard_name"
          fi
          ((i++)) || true
        done
      else
        # Single file
        url=$(resolve_hf_url "$download_url" "$gguf")
        if [[ -n "$url" ]]; then
          download_one "$url" "$dest_dir" "$gguf"
        else
          echo "  skip: could not resolve URL for $gguf"
        fi
      fi
    fi
    # Extra required files (e.g. mmproj for vision; or download_extra list in YAML). Same repo.
    if [[ -n "$extras" ]]; then
      IFS=',' read -ra extra_files <<< "$extras"
      for outname in "${extra_files[@]}"; do
        outname=$(echo "$outname" | tr -d ' ')
        [[ -z "$outname" ]] && continue
        url=$(resolve_hf_url "$download_url" "$outname")
        if [[ -n "$url" ]]; then
          download_one "$url" "$dest_dir" "$outname"
        else
          echo "  skip: could not resolve URL for $outname"
        fi
      done
    fi
  fi

  # ---- vLLM safetensors: only when backend is vllm — list repo files, download each with aria2 ----
  if [[ "$backend" == "vllm" ]] && [[ -n "$vllm_download_repo" ]]; then
    vllm_dir="$dest_dir/vllm"
    if [[ -d "$vllm_dir" ]] && [[ -f "$vllm_dir/config.json" ]]; then
      echo "  skip (vLLM dir exists): $vllm_dir"
    else
      vllm_repo_url="https://huggingface.co/${vllm_download_repo}"
      echo "  download vLLM model (aria2): $vllm_download_repo -> $vllm_dir"
      while IFS= read -r path || [[ -n "$path" ]]; do
        [[ -z "$path" ]] && continue
        url=$(resolve_hf_url "$vllm_repo_url" "$path")
        [[ -z "$url" ]] && continue
        if [[ "$path" == */* ]]; then
          file_dir="$vllm_dir/$(dirname "$path")"
          outname=$(basename "$path")
        else
          file_dir="$vllm_dir"
          outname="$path"
        fi
        download_one "$url" "$file_dir" "$outname"
      done < <(list_hf_repo_files "$vllm_download_repo")
    fi
  fi
  echo ""
done < <(get_model_keys "$@")

echo "Done. Models under $ROOT/models/"
