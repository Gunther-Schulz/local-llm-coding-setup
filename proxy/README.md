# LLM proxy

Forwards requests to one or two backends. Used so Cursor talks to a single endpoint (proxy port); the proxy routes by content (e.g. image → vision server, else coding server).

## Layout

- **config.py** – Load `config/server.env` and env; build `ProxyConfig` (single backend or Code + Vision URLs).
- **router.py** – Detect image in request body (OpenAI-style `messages[].content` with `type: "image_url"` / `"image"`); choose backend URL.
- **forward.py** – HTTP forward: send request to backend, stream response back to client.
- **server.py** – `HTTPServer` handler: route via `router.get_backend_url`, then `forward.forward`.
- **__main__.py** – Entry: load config, run server. `python -m proxy [--debug]`.

## Modes

- **Single:** `BACKEND_URL` (default `http://HOST:8001`). All requests go to that URL.
- **Code + Vision:** `CODE_VISION_VISION_PORT` (8002) and `CODE_VISION_CODING_PORT` (8001) set in env. POST `/v1/chat/completions` with an image in the body → vision server; all other requests → coding server.

## Run

From repo root: `./start-proxy.sh [--debug]`. Start the backend(s) first (e.g. `./run_code_vision.sh` for Mode 4, or `./run_coding.sh` for Mode 2).
