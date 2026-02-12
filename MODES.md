# Usage modes

Only one mode is active at a time. Configure each mode in **config/server.env**; then run the matching launcher.

| Mode | Launcher | What runs | Client points at |
|------|----------|-----------|------------------|
| **1. Pure chat** | `./run_chat.sh` | One LLM on 8001 | `http://HOST:8001/v1` or proxy (if `PURE_CHAT_PROXY_PORT` set) |
| **2. Coding** | `./run_coding.sh` | One LLM on 8001 + proxy on 8010 | Cursor → proxy (8010) → backend (8001) |
| **3. Notebook LM** | `./run_notebook.sh` | Embedding on 8002, chat on 8001 | Open Notebook: embedding 8002, chat 8001 |

## Core

- **run_server.sh** – Starts one llama-server. Used by all launchers. No default model.  
  `./run_server.sh MODEL_KEY [PORT]`  
  With no args, prints usage and exits. Use a launcher or pass model (and optional port) explicitly.

## Launchers

- **run_chat.sh** – Mode 1. Starts one server from `PURE_CHAT_MODEL` / `PURE_CHAT_PORT`.
- **run_coding.sh** – Mode 2. Starts coding LLM only (no proxy yet). When proxy is ready, run `start-proxy.sh` separately with `BACKEND_URL=http://HOST:8001`.
- **run_notebook.sh** – Mode 3. Starts embedding server (8002), then chat server (8001). Both stay running in background; script exits when both are up.

## Proxy

- **start-proxy.sh** – Tool proxy. With no args, uses `config/server.env` and env `BACKEND_URL` (default `http://HOST:8001`), `PROXY_PORT` (default 8010).  
  Launchers set these when they call the proxy (e.g. run_coding.sh sets `BACKEND_URL` and `PROXY_PORT`).

## Chat CLI

- **scripts/chat.py** – Uses `PURE_CHAT_PORT` (or `PORT`) and `HOST` from `config/server.env` for the base URL.

## Config

- **config/server.env** – Per-mode model and port vars. Proxy backend is always the **chat** server on **8001** so the proxy can point at one port for all modes.
