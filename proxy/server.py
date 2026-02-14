"""HTTP server handler: route request to backend, then forward."""
from __future__ import annotations

from http.server import HTTPServer, BaseHTTPRequestHandler

from .config import ProxyConfig, load_config
from .router import get_backend_url
from .forward import forward, log


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
            backend_url = get_backend_url(
                method, self.path, body, config
            )
            forward(
                self,
                backend_url,
                method,
                self.path,
                dict(self.headers),
                body,
                debug=debug,
            )

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
