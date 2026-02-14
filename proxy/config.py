"""Load proxy configuration from config/server.env and environment."""
from __future__ import annotations

import os
from pathlib import Path
from dataclasses import dataclass

# Repo root (parent of proxy/)
ROOT = Path(__file__).resolve().parent.parent
SERVER_ENV = ROOT / "config" / "server.env"


@dataclass
class ProxyConfig:
    """Proxy configuration: single backend or Code + Vision (vision + coding backends)."""
    mode: str  # "single" | "code_vision"
    backend_url: str
    proxy_port: int
    vision_url: str | None = None
    coding_url: str | None = None

    @property
    def is_code_vision(self) -> bool:
        return self.mode == "code_vision"


def _load_server_env() -> None:
    if not SERVER_ENV.exists():
        return
    for line in SERVER_ENV.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key and value and key.isidentifier():
            os.environ.setdefault(key, value)


def load_config() -> ProxyConfig:
    """Load config from server.env and env. Prefer env vars over file."""
    _load_server_env()

    host = os.environ.get("HOST", "127.0.0.1")
    proxy_port = int(os.environ.get("PROXY_PORT", "8010"))

    # Code + Vision mode: two backends (vision port, coding port)
    vision_port = os.environ.get("CODE_VISION_VISION_PORT")
    coding_port = os.environ.get("CODE_VISION_CODING_PORT")
    if vision_port and coding_port:
        mode = "code_vision"
        vision_url = f"http://{host}:{vision_port}".rstrip("/")
        coding_url = f"http://{host}:{coding_port}".rstrip("/")
        # Default backend for non-chat or when no image: coding
        backend_url = os.environ.get("BACKEND_URL", coding_url).strip().rstrip("/")
        return ProxyConfig(
            mode=mode,
            backend_url=backend_url,
            proxy_port=proxy_port,
            vision_url=vision_url,
            coding_url=coding_url,
        )

    # Single backend mode
    backend_url = os.environ.get("BACKEND_URL", f"http://{host}:8001").strip().rstrip("/")
    return ProxyConfig(
        mode="single",
        backend_url=backend_url,
        proxy_port=proxy_port,
        vision_url=None,
        coding_url=None,
    )
