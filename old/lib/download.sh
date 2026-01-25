#!/bin/bash
# Single download implementation: aria2c > hf-cli > wget > curl
# Reused by model-selector (LLM) and vision-manager (vision gguf + mmproj).
#
# download_file URL DEST_PATH [MIN_BYTES]
#   MIN_BYTES: optional; if set and file exists, consider complete only when
#   size >= MIN_BYTES and no .aria2. Default 0: consider complete if file exists and no .aria2.
# Returns 0 on success, 1 on failure.

download_file() {
    local url="$1" dest="$2" min_bytes="${3:-0}"
    local dir=$(dirname "$dest") out=$(basename "$dest")

    # Already complete?
    if [[ -f "$dest" && ! -f "${dest}.aria2" ]]; then
        if [[ -n "$min_bytes" && "$min_bytes" -gt 0 ]]; then
            local size=$(stat -c%s "$dest" 2>/dev/null || echo "0")
            if [[ $size -ge $min_bytes ]]; then
                return 0
            fi
            rm -f "$dest" "${dest}.aria2"
        else
            return 0
        fi
    fi

    [[ -z "$url" || "$url" == "none" ]] && { echo "ERROR: No URL" >&2; return 1; }
    mkdir -p "$dir"

    if command -v aria2c &>/dev/null; then
        [[ -f "${dest}.aria2" ]] && echo "Resuming..." >&2
        aria2c --continue=true --max-connection-per-server=16 --min-split-size=1M --split=16 \
            --file-allocation=none --console-log-level=warn --summary-interval=0 \
            -d "$dir" -o "$out" "$url" >&2
        [[ -f "${dest}.aria2" ]] && rm -f "${dest}.aria2"
    elif command -v huggingface-cli &>/dev/null && [[ "$url" =~ huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$ ]]; then
        local repo="${BASH_REMATCH[1]}" filename="${BASH_REMATCH[2]}"
        HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download "$repo" "$filename" \
            --local-dir "$dir" --local-dir-use-symlinks False >&2
        [[ -f "$dir/$filename" && "$dir/$filename" != "$dest" ]] && mv "$dir/$filename" "$dest"
    elif command -v wget &>/dev/null; then
        wget -c -O "$dest" "$url" >&2
    elif command -v curl &>/dev/null; then
        curl -L -C - -o "$dest" "$url" >&2
    else
        echo "ERROR: No download tool (aria2c, huggingface-cli, wget, curl)" >&2
        return 1
    fi

    [[ -f "$dest" ]] && return 0
    return 1
}
