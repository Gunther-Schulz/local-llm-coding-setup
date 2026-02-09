#!/usr/bin/env bash
# Download Qwen3-Coder-Next Q2, Q3, Q4 GGUF from Unsloth (HuggingFace).
# Activates conda env "vLLM" if conda is available. Uses aria2 for downloads when present.
set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/../.." && pwd)"
REPO="unsloth/Qwen3-Coder-Next-GGUF"
REPO_API="https://huggingface.co/api/models/${REPO}/tree/main"
RESOLVE_URL="https://huggingface.co/${REPO}/resolve/main"

# Activate conda env "vLLM" if conda is available
if command -v conda &>/dev/null; then
  _conda_base="$(conda info --base 2>/dev/null)"
  if [[ -n "$_conda_base" && -f "${_conda_base}/etc/profile.d/conda.sh" ]]; then
    set +e
    source "${_conda_base}/etc/profile.d/conda.sh"
    conda activate vLLM 2>/dev/null
    set -e
  fi
fi

cd "$ROOT"

# Fetch file list from Hugging Face API
echo "Fetching file list from ${REPO}..."
_filelist=""
if command -v curl &>/dev/null; then
  _filelist="$(curl -sL "$REPO_API")"
fi
_use_aria2=false
if command -v aria2c &>/dev/null; then
  _use_aria2=true
fi

# aria2 options aligned with stack/download.py (resume, 16 conn/split, no prealloc)
download_with_aria2() {
  local url="$1"
  local dir="$2"
  local outname="$3"
  mkdir -p "$dir"
  aria2c --continue=true --max-connection-per-server=16 --min-split-size=1M --split=16 \
    --file-allocation=none -d "$dir" -o "$outname" --allow-overwrite=true --console-log-level=warn "$url"
}

download_with_hf() {
  local pattern="$1"
  local dir="$2"
  mkdir -p "$dir"
  if command -v huggingface-cli &>/dev/null; then
    huggingface-cli download "$REPO" --include "$pattern" --local-dir "$dir"
  else
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$REPO', local_dir='$dir', allow_patterns='$pattern')
"
  fi
}

# Prefer UD (Unsloth) filenames so scenarios.cfg paths match
for quant in q2 q3 q4; do
  case "$quant" in
    q2) pattern="*UD*Q2_K*";   dir="models/qwen3-coder-next-q2";;
    q3) pattern="*UD*Q3_K_M*"; dir="models/qwen3-coder-next-q3";;
    q4) pattern="*UD*Q4_K_XL*"; dir="models/qwen3-coder-next-q4";;
  esac
  echo "Downloading $quant -> $dir (pattern: $pattern)"

  # Prefer UD (Unsloth) filenames so scenarios.cfg paths match
  _path=""
  if [[ -n "$_filelist" ]]; then
    _path=$(echo "$_filelist" | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    q = '${quant}'
    candidates = []
    for o in data:
        p = o.get('path') or o.get('rfilename', '')
        if not p or not p.endswith('.gguf'):
            continue
        if q == 'q2' and 'Q2_K' in p: candidates.append(p)
        elif q == 'q3' and 'Q3_K_M' in p: candidates.append(p)
        elif q == 'q4' and 'Q4_K_XL' in p: candidates.append(p)
    # Prefer path containing UD (matches scenarios.cfg)
    for c in candidates:
        if 'UD' in c:
            print(c); sys.exit(0)
    if candidates:
        print(candidates[0]); sys.exit(0)
except Exception:
    pass
" 2>/dev/null)
  fi

  if [[ -n "$_path" && "$_use_aria2" == true ]]; then
    _url="${RESOLVE_URL}/${_path}"
    _base=$(basename "$_path")
    download_with_aria2 "$_url" "$ROOT/$dir" "$_base"
  else
    download_with_hf "$pattern" "$dir"
  fi
done

echo "Done. Check models/qwen3-coder-next-{q2,q3,q4}/ for .gguf files."
echo "If filenames differ from scenarios.cfg, edit scenarios.cfg model_path."
