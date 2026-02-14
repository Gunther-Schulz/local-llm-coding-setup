#!/usr/bin/env bash
# Download all models from config/models/*.yaml using aria2 (resume on interrupt, skip existing).
# Requires: aria2c, curl, python3, PyYAML. Usage: ./scripts/download-models.sh [MODEL_KEY ...]
#   No args = download all models that have a download_url. Optional args = only those model_key(s).
#
# Per-model YAML: download_url (repo or direct .gguf link). gguf (and mmproj) from top-level or llama:.
# If llama.mmproj is set, it is downloaded from the same repo. Optional download_extra: [ "file.gguf", ... ]
# for any other required files from the same repo (e.g. vision mmproj, tokenizer, etc.).

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

# Read download_url, gguf, and extra required files from a model YAML.
# Output: download_url|gguf|extra1,extra2,... (extras = mmproj if set, plus download_extra list).
# gguf/mmproj can be top-level or under llama:.
read_model_download() {
  python3 - "$1" << 'PY'
import sys, yaml
path = sys.argv[1]
with open(path) as f:
    data = yaml.safe_load(f) or {}
url = (data.get("download_url") or "").strip()
if not url or url.lower() == "none":
    sys.exit(0)
llama = data.get("llama") if isinstance(data.get("llama"), dict) else {}
gguf = (data.get("gguf") or llama.get("gguf") or "").strip()
extras = []
if llama.get("mmproj"):
    extras.append(str(llama["mmproj"]).strip())
for f in data.get("download_extra") or []:
    if f and str(f).strip() and str(f).strip() not in extras:
        extras.append(str(f).strip())
extras_str = ",".join(extras)
print(f"{url}|{gguf}|{extras_str}")
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

while read -r model_key; do
  [[ -z "$model_key" ]] && continue
  yaml_file="$MODELS_DIR/${model_key}.yaml"
  if [[ ! -f "$yaml_file" ]]; then
    continue
  fi
  line=$(read_model_download "$yaml_file") || true
  [[ -z "$line" ]] && echo "Model: $model_key (no download_url), skip" && echo "" && continue
  IFS='|' read -r download_url gguf extras <<< "$line"
  dest_dir="$ROOT/models/${model_key}"
  echo "Model: $model_key -> $dest_dir"

  if [[ "$download_url" == *"/resolve/main/"* && "$download_url" == *.gguf ]]; then
    # Direct .gguf link: download that file only.
    outname=$(basename "$download_url")
    download_one "$download_url" "$dest_dir" "$outname"
    expected="$gguf"
    if [[ -n "$expected" && "$outname" != "$expected" && -f "$dest_dir/$outname" && ! -f "$dest_dir/$expected" ]]; then
      ln -sf "$outname" "$dest_dir/$expected" 2>/dev/null || true
    fi
  else
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
  echo ""
done < <(get_model_keys "$@")

echo "Done. Models under $ROOT/models/"
