"""Forward an HTTP request to a backend and stream the response back to the client."""
from __future__ import annotations

import json
import socket
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import urlparse


def _redact(h: dict) -> dict:
    return {
        k: "***" if k.lower() in ("authorization", "api-key", "x-api-key") else v
        for k, v in h.items()
    }


def log(msg: str, *args: object) -> None:
    sys.stderr.write("[proxy] " + (msg % args if args else msg) + "\n")
    sys.stderr.flush()


def forward(
    handler: BaseHTTPRequestHandler,
    backend_url: str,
    method: str,
    path: str,
    headers: dict,
    body: bytes | None,
    *,
    debug: bool = False,
) -> None:
    """
    Forward the request to backend_url and stream the response back to the client.
    """
    parsed = urlparse(backend_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    req_headers = dict(headers)
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

    if debug:
        log(">>> %s %s -> %s", method, path, backend_url)
        log(">>> REQUEST HEADERS: %s", json.dumps(_redact(req_headers), sort_keys=True))
        if body:
            log(">>> REQUEST BODY length: %d bytes", len(body))
            try:
                log(">>> REQUEST BODY (preview): %s", body[:500].decode("utf-8", errors="replace"))
            except Exception:
                pass

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1800)
        sock.connect((host, port))
        sock.sendall(req_bytes)

        buf = b""
        while b"\r\n\r\n" not in buf and b"\n\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        if not buf:
            handler.send_error(502, "Backend closed")
            return

        sep = b"\r\n\r\n" if b"\r\n\r\n" in buf else b"\n\n"
        head_part, _, _ = buf.partition(sep)
        first_line = (
            head_part.split(b"\r\n")[0]
            if b"\r\n" in head_part
            else head_part.split(b"\n")[0]
        )

        resp_headers = {}
        for line in head_part.split(b"\r\n" if b"\r\n" in head_part else b"\n")[1:]:
            if b":" in line:
                k, _, v = line.partition(b":")
                resp_headers[k.strip().decode("latin-1")] = v.strip().decode("latin-1")

        if debug:
            log("<<< %s", first_line.decode("latin-1"))
            log("<<< RESPONSE HEADERS: %s", json.dumps(_redact(resp_headers), sort_keys=True))

        handler.wfile.write(head_part + sep)
        handler.wfile.flush()

        chunk_num = 0
        while True:
            chunk = sock.recv(8192)
            if not chunk:
                break
            chunk_num += 1
            if debug and chunk_num == 1:
                log("<<< FIRST BODY CHUNK: %d bytes (%r)", len(chunk), chunk[:200])
            try:
                handler.wfile.write(chunk)
                handler.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                if debug:
                    log("<<< CLIENT DISCONNECTED after %d body chunks", chunk_num)
                break
        sock.close()

    except (socket.timeout, OSError, ConnectionRefusedError) as e:
        log("<<< ERROR: %s", e)
        handler.send_error(502, str(e))
