# LLM proxy

Forwards requests to one or two backends. Used so Cursor talks to a single endpoint (proxy port); the proxy routes by content (e.g. image → vision server, else coding server).

## Layout

- **config.py** – Load `config/server.env` and env; build `ProxyConfig` (single backend or Code + Vision URLs).
- **router.py** – Detect image in request body (OpenAI-style `messages[].content` with `type: "image_url"` / `"image"`); choose backend URL.
- **forward.py** – HTTP forward: send request to backend, stream response back to client.
- **server.py** – `HTTPServer` handler: route via `router.get_backend_url`, then `forward.forward`. When image + Code+Vision: two-step via **vision_step**.
- **vision_step.py** – Two-step Code+Vision: call vision backend with image + “describe” prompt (no tools); get text description; build new request with description instead of image; forward to coding backend and stream that response.
- **__main__.py** – Entry: load config, run server. `python -m proxy [--debug]`.

## Modes

- **Single:** `BACKEND_URL` (default `http://HOST:8001`). All requests go to that URL.
- **Code + Vision (two-step):** `CODE_VISION_VISION_PORT` (8002) and `CODE_VISION_CODING_PORT` (8001). POST `/v1/chat/completions` **with an image**: (1) proxy sends image + simple “describe” prompt to vision (no tools); (2) proxy replaces the image in the request with the vision description and sends to coding (with tools); (3) coding response is streamed to the client. So Cursor gets the **coding** model’s reply, using the image description. Requests without an image go straight to the coding server.

## Run

From repo root: `./start-proxy.sh [--debug]`. Start the backend(s) first (e.g. `./run_code_vision.sh` for Mode 4, or `./run_coding.sh` for Mode 2).
