# Usage modes

Only one mode is active at a time. Configure each mode in **config/server.env**; then run the matching launcher.

| Mode | Launcher | What runs | Client points at |
|------|----------|-----------|------------------|
| **1. Pure chat** | `./run_chat.sh` | One LLM on 8001 | `http://HOST:8001/v1` or proxy (if `PURE_CHAT_PROXY_PORT` set) |
| **2. Coding** | `./run_coding.sh` | One LLM on 8001 + proxy on 8010 | Cursor → proxy (8010) → backend (8001) |
| **3. Notebook LM** | `./run_notebook.sh` | One llama-server (router mode) on 8001 | Same port: `model=bge-m3` for embeddings, `model=notebook-chat` for chat |
| **4. Code + Vision** | `./run_code_vision.sh` | Vision on 8002 + coding on 8001 (like Mode 3: two models) | Vision: `http://HOST:8002/v1`; coding: `http://HOST:8001/v1` (proxy can combine: image → vision, then coding) |

## Core

- **run_server.sh** – Starts one llama-server. Used by all launchers. No default model.  
  `./run_server.sh MODEL_KEY [PORT]`  
  With no args, prints usage and exits. Use a launcher or pass model (and optional port) explicitly.

## Launchers

- **run_chat.sh** – Mode 1. Starts one server from `PURE_CHAT_MODEL` / `PURE_CHAT_PORT`.
- **run_coding.sh** – Mode 2. Starts coding LLM only (no proxy yet). When proxy is ready, run `start-proxy.sh` separately with `BACKEND_URL=http://HOST:8001`.
- **run_notebook.sh** – Mode 3. One llama-server in router mode: `--models-dir` with embedding + chat (symlinks from `scripts/build_notebook_router_dir.sh`), plus `--models-preset config/notebook-router-models.ini` so the bge-m3 child is started with `--embeddings`. Single port; use `model=bge-m3` for `/v1/embeddings` and `model=notebook-chat` for `/v1/chat/completions`. LRU eviction when `--models-max` (2) is reached.
- **run_code_vision.sh** – Mode 4. Two models (like Mode 3: embedding + chat). Vision server from `CODE_VISION_VISION_MODEL` on `CODE_VISION_VISION_PORT` (8002); coding server from `CODING_MODEL` on `CODE_VISION_CODING_PORT` (8001). Vision = pure image→text; coding = chat/tools (Cursor main model). When Cursor sends chat + image, use vision for the image, then coding with that context.

## Proxy

- **start-proxy.sh** – Tool proxy. With no args, uses `config/server.env` and env `BACKEND_URL` (default `http://HOST:8001`), `PROXY_PORT` (default 8010).  
  Launchers set these when they call the proxy (e.g. run_coding.sh sets `BACKEND_URL` and `PROXY_PORT`).

## Chat CLI

- **scripts/chat.py** – Uses `PURE_CHAT_PORT` (or `PORT`) and `HOST` from `config/server.env` for the base URL.

## Config

- **config/server.env** – Per-mode model and port vars. Proxy backend is always the **chat** server on **8001** so the proxy can point at one port for all modes.
