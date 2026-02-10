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

# MXFP4_MOE first (priority); then others. Prefer UD (Unsloth) filenames so scenarios.cfg paths match.
for quant in mxfp4 q2 iq3xxs q3s q3 q4 q4m; do
  case "$quant" in
    q2)     pattern="*UD*Q2_K*";    dir="models/qwen3-coder-next-q2";;
    iq3xxs) pattern="*UD*IQ3_XXS*"; dir="models/qwen3-coder-next-iq3xxs";;
    q3s)    pattern="*Q3_K_S*";     dir="models/qwen3-coder-next-q3s";;
    q3)     pattern="*UD*Q3_K_M*";  dir="models/qwen3-coder-next-q3";;
    q4)     pattern="*UD*Q4_K_XL*"; dir="models/qwen3-coder-next-q4";;
    q4m)    pattern="*Q4_K_M*";    dir="models/qwen3-coder-next-q4m";;
    mxfp4)  pattern="*MXFP4_MOE*"; dir="models/qwen3-coder-next-mxfp4";;
  esac
  echo "Downloading $quant -> $dir (pattern: $pattern)"

  # Prefer UD (Unsloth) filenames so scenarios.cfg paths match; exact-name quants use first match
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
        elif q == 'iq3xxs' and 'IQ3_XXS' in p: candidates.append(p)
        elif q == 'q3s' and 'Q3_K_S' in p: candidates.append(p)
        elif q == 'q3' and 'Q3_K_M' in p: candidates.append(p)
        elif q == 'q4' and 'Q4_K_XL' in p: candidates.append(p)
        elif q == 'q4m' and 'Q4_K_M' in p and 'Q4_K_XL' not in p: candidates.append(p)
        elif q == 'mxfp4' and 'MXFP4_MOE' in p: candidates.append(p)
    for c in candidates:
        if q in ('q4m', 'mxfp4', 'iq3xxs', 'q3s'):
            print(c); sys.exit(0)
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

echo "Done. Check models/qwen3-coder-next-{q2,iq3xxs,q3s,q3,q4,q4m,mxfp4}/ for .gguf files."

# Qwen3-Coder-30B-A3B UD-Q4_K_XL (unsloth) – highest coding quality that fits full GPU (~17 GB, 147K context)
Q30_REPO="unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF"
Q30_DIR="models/qwen3-coder-30b-a3b-q4_k_xl"
Q30_FILE="Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf"
echo ""
echo "Downloading Qwen3-Coder-30B-A3B UD-Q4_K_XL -> $Q30_DIR"
mkdir -p "$ROOT/$Q30_DIR"
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "$Q30_REPO" "$Q30_FILE" --local-dir "$ROOT/$Q30_DIR"
else
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$Q30_REPO', filename='$Q30_FILE', local_dir='$ROOT/$Q30_DIR')
"
fi
echo "Done. Check $Q30_DIR for .gguf file."

# GLM-4.7-Flash MXFP4_MOE (noctrex) – fits 5090 full GPU, coding comparison
GLM_REPO="noctrex/GLM-4.7-Flash-MXFP4_MOE-GGUF"
GLM_DIR="models/glm-4.7-flash-mxfp4"
GLM_FILE="GLM-4.7-Flash-MXFP4_MOE.gguf"
echo ""
echo "Downloading GLM-4.7-Flash MXFP4_MOE -> $GLM_DIR"
mkdir -p "$ROOT/$GLM_DIR"
if command -v huggingface-cli &>/dev/null; then
  huggingface-cli download "$GLM_REPO" "$GLM_FILE" --local-dir "$ROOT/$GLM_DIR"
else
  python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='$GLM_REPO', filename='$GLM_FILE', local_dir='$ROOT/$GLM_DIR')
"
fi
echo "Done. Check $GLM_DIR for .gguf file."

echo ""
echo "If filenames differ from scenarios.cfg, edit scenarios.cfg model_path."
