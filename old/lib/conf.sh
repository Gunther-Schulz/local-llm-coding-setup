#!/bin/bash
# Generic pipe-delimited config reader
# Used by model-selector and vision-manager to avoid duplicated MODELS_CONF/VISION_MODELS_CONF paths
# and grep/while-read patterns.
#
# Usage: source this file after ROOT is set. conf_path/conf_get/conf_iter use:
#   models -> $ROOT/models.conf (or $MODELS_CONF)
#   vision -> $ROOT/vision-models.conf (or $VISION_MODELS_CONF)

# Resolve config file path by name
# conf_path "models" | "vision"
conf_path() {
    case "$1" in
        models)  echo "${MODELS_CONF:-$ROOT/models.conf}" ;;
        vision)  echo "${VISION_MODELS_CONF:-$ROOT/vision-models.conf}" ;;
        *)       echo "ERROR: unknown conf name: $1" >&2; return 1 ;;
    esac
}

# Get one line by key (first column). Empty if not found.
# conf_get "models" "qwen3-30b-q2"
conf_get() {
    local name="$1" key="$2"
    local f
    f=$(conf_path "$name") || return 1
    [[ -f "$f" ]] || return 1
    grep "^${key}|" "$f" | head -1
}

# Stream non-comment, non-empty lines. No trailing newline guarantee.
# conf_iter "models" | while IFS='|' read -r ...; do
conf_iter() {
    local name="$1"
    local f
    f=$(conf_path "$name") || return 1
    [[ -f "$f" ]] || return 1
    while IFS= read -r line || [[ -n "$line" ]]; do
        [[ "$line" =~ ^#.*$ ]] && continue
        [[ -z "$line" ]] && continue
        echo "$line"
    done < "$f"
}
