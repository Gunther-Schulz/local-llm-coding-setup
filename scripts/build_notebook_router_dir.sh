#!/usr/bin/env bash
# Build models/.notebook-router with symlinks so llama-server router mode can load
# embedding + chat from config (EMBEDDING_MODEL, NOTEBOOK_CHAT_MODEL).
# Usage: scripts/build_notebook_router_dir.sh
# Requires: config/server.env and config/models/<key>.yaml for embedding and chat keys.

set -e
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
[[ -f "$ROOT/config/server.env" ]] || { echo "config/server.env not found" >&2; exit 1; }
set -a
source "$ROOT/config/server.env"
set +a

ROUTER_DIR="$ROOT/models/.notebook-router"
mkdir -p "$ROUTER_DIR"

# Resolve embedding model GGUF path
eval "$("$ROOT/scripts/load_model_config.sh" "${EMBEDDING_MODEL:?EMBEDDING_MODEL not set}")"
EMBED_PATH="$ROOT/models/${EMBEDDING_MODEL}/${GGUF:?}"
[[ -f "$EMBED_PATH" ]] || { echo "Embedding model not found: $EMBED_PATH" >&2; exit 1; }

# Resolve chat model GGUF path (may be first shard of multi-file)
eval "$("$ROOT/scripts/load_model_config.sh" "${NOTEBOOK_CHAT_MODEL:?NOTEBOOK_CHAT_MODEL not set}")"
CHAT_PATH="$ROOT/models/${NOTEBOOK_CHAT_MODEL}/${GGUF:?}"
[[ -f "$CHAT_PATH" ]] || { echo "Chat model not found: $CHAT_PATH" >&2; exit 1; }

# Symlink with fixed names so API model names are predictable (targets absolute for portability)
ln -sf "$EMBED_PATH" "$ROUTER_DIR/bge-m3.gguf"
ln -sf "$CHAT_PATH" "$ROUTER_DIR/notebook-chat.gguf"

echo "Router dir: $ROUTER_DIR (bge-m3.gguf, notebook-chat.gguf)"
echo "API model names: bge-m3, notebook-chat"
