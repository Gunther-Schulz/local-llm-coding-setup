#!/usr/bin/env python3
"""
Minimal proxy: forwards requests to the backend and logs request/response.
No other logic. Pass-through so the client gets exactly what the backend sends.

Env: BACKEND_URL (default http://127.0.0.1:8001), PROXY_PORT (8010).
Usage: ./start-proxy.sh  or  python scripts/chat_proxy.py
"""
from __future__ import annotations

import json
import os
import socket
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001").strip().rstrip("/")
PORT = int(os.environ.get("PROXY_PORT", "8010"))

def _redact(h: dict) -> dict:
    return {k: "***" if k.lower() in ("authorization", "api-key", "x-api-key") else v for k, v in h.items()}

def _log(msg: str, *args: object) -> None:
    sys.stderr.write("[proxy] " + (msg % args if args else msg) + "\n")
    sys.stderr.flush()

class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        pass  # we log ourselves

    def do_GET(self) -> None:
        self._forward("GET", None)

    def do_POST(self) -> None:
        cl = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(cl) if cl else b""
        self._forward("POST", body)

    def do_OPTIONS(self) -> None:
        self._forward("OPTIONS", None)

    def do_HEAD(self) -> None:
        self._forward("HEAD", None)

    def _forward(self, method: str, body: bytes | None) -> None:
        parsed = urlparse(BACKEND_URL)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)

        # Build request for backend
        path = self.path
        req_headers = dict(self.headers)
        req_headers["Host"] = f"{host}:{port}" if port not in (80, 443) else host
        if body is not None:
            req_headers["Content-Length"] = str(len(body))
        req_headers.pop("Connection", None)

        req_lines = [f"{method} {path} HTTP/1.1"]
        for k, v in req_headers.items():
            req_lines.append(f"{k}: {v}")
        req_bytes = "\r\n".join(req_lines).encode("latin-1") + b"\r\n\r\n"
        if body:
            req_bytes += body

        # Log request
        _log(">>> %s %s", method, path)
        _log(">>> REQUEST HEADERS: %s", json.dumps(_redact(req_headers), sort_keys=True))
        if body:
            _log(">>> REQUEST BODY length: %d bytes", len(body))
            try:
                _log(">>> REQUEST BODY (preview): %s", body[:500].decode("utf-8", errors="replace"))
            except Exception:
                pass

        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1800)
            sock.connect((host, port))
            sock.sendall(req_bytes)

            # Read response: status + headers (until \r\n\r\n)
            buf = b""
            while b"\r\n\r\n" not in buf and b"\n\n" not in buf:
                chunk = sock.recv(4096)
                if not chunk:
                    break
                buf += chunk
            if not buf:
                self.send_error(502, "Backend closed")
                return

            # Send headers immediately to client, then stream body
            sep = b"\r\n\r\n" if b"\r\n\r\n" in buf else b"\n\n"
            head_part, _, _ = buf.partition(sep)
            first_line = head_part.split(b"\r\n")[0] if b"\r\n" in head_part else head_part.split(b"\n")[0]

            # Log and send headers
            resp_headers = {}
            for line in head_part.split(b"\r\n" if b"\r\n" in head_part else b"\n")[1:]:
                if b":" in line:
                    k, _, v = line.partition(b":")
                    resp_headers[k.strip().decode("latin-1")] = v.strip().decode("latin-1")
            _log("<<< %s", first_line.decode("latin-1"))
            _log("<<< RESPONSE HEADERS: %s", json.dumps(_redact(resp_headers), sort_keys=True))
            _log("<<< SENDING %d bytes (headers) to client...", len(head_part) + len(sep))
            self.wfile.write(head_part + sep)
            self.wfile.flush()

            # Stream body as we read it
            chunk_num = 0
            while True:
                chunk = sock.recv(8192)
                if not chunk:
                    break
                chunk_num += 1
                if chunk_num == 1:
                    _log("<<< FIRST BODY CHUNK: %d bytes (%r)", len(chunk), chunk[:200])
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    _log("<<< CLIENT DISCONNECTED after %d body chunks", chunk_num)
                    break
            sock.close()

        except (socket.timeout, OSError, ConnectionRefusedError) as e:
            _log("<<< ERROR: %s", e)
            self.send_error(502, str(e))


def main() -> None:
    global BACKEND_URL, PORT
    BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001").strip().rstrip("/")
    PORT = int(os.environ.get("PROXY_PORT", "8010"))
    _log("Proxy: %s -> http://0.0.0.0:%s (log only)", BACKEND_URL, PORT)
    server = HTTPServer(("0.0.0.0", PORT), _Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
