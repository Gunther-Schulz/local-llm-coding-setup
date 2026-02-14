"""HTTP server handler: route request to backend, then forward."""
from __future__ import annotations

from http.server import HTTPServer, BaseHTTPRequestHandler

from .config import ProxyConfig, load_config
from .router import get_backend_url, request_has_image
from .forward import forward, log
from .vision_step import get_vision_description, build_coding_body_with_description


def make_handler(config: ProxyConfig, debug: bool = False) -> type[BaseHTTPRequestHandler]:
    """Build a request handler class that uses the given config."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass  # we log ourselves

        def do_GET(self) -> None:
            self._handle("GET", None)

        def do_POST(self) -> None:
            cl = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(cl) if cl else b""
            self._handle("POST", body)

        def do_OPTIONS(self) -> None:
            self._handle("OPTIONS", None)

        def do_HEAD(self) -> None:
            self._handle("HEAD", None)

        def _handle(self, method: str, body: bytes | None) -> None:
            headers = dict(self.headers)
            path = self.path

            # Two-step Code+Vision: image request → vision (describe, no tools) → coding (with description)
            if (
                method == "POST"
                and path.rstrip("/").endswith("/v1/chat/completions")
                and body
                and config.is_code_vision
                and config.vision_url
                and config.coding_url
                and request_has_image(body)
            ):
                description, user_text = get_vision_description(
                    config.vision_url, body, headers, debug=debug
                )
                new_body = build_coding_body_with_description(body, description, user_text)
                if debug:
                    log("vision_step: description len=%d -> coding", len(description))
                forward(
                    self,
                    config.coding_url,
                    method,
                    path,
                    headers,
                    new_body,
                    debug=debug,
                )
                return

            backend_url = get_backend_url(method, path, body, config)
            forward(self, backend_url, method, path, headers, body, debug=debug)

    return _Handler


def run_server(config: ProxyConfig | None = None, debug: bool = False) -> None:
    """Run the proxy HTTP server. Load config if not provided."""
    if config is None:
        config = load_config()
    log(
        "Proxy: %s -> http://0.0.0.0:%s (mode=%s)",
        config.backend_url if config.mode == "single" else f"vision={config.vision_url} coding={config.coding_url}",
        config.proxy_port,
        config.mode,
    )
    handler = make_handler(config, debug=debug)
    server = HTTPServer(("0.0.0.0", config.proxy_port), handler)
    server.serve_forever()
