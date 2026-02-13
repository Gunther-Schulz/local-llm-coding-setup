#!/usr/bin/env python3
"""
Thin proxy in front of llama-server for coding (Qwen3 Coder).
- Forwards all requests to BACKEND_URL (e.g. http://127.0.0.1:8001).
- For POST /v1/chat/completions: if body has "tools" and tool_choice is missing or "auto",
  sets tool_choice to "required" before forwarding (avoids client-side control).
Env: BACKEND_URL (default http://127.0.0.1:8001), PROXY_PORT (default 8010).
Usage: BACKEND_URL=http://127.0.0.1:8001 PROXY_PORT=8010 ./scripts/chat_proxy.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")
PORT = int(os.environ.get("PROXY_PORT", "8010"))


def apply_tool_choice(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body
    tools = data.get("tools")
    if not tools or not isinstance(tools, list) or len(tools) == 0:
        return body
    choice = data.get("tool_choice")
    if choice == "required":
        return body
    if choice == "none":
        return body
    # "auto" or missing -> force "required" for Qwen3 Coder
    data["tool_choice"] = "required"
    return json.dumps(data).encode("utf-8")


class ProxyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._proxy(method="GET", body=None)

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b""
        if self.path == "/v1/chat/completions" and body:
            body = apply_tool_choice(body)
        self._proxy(method="POST", body=body)

    def do_OPTIONS(self):
        self._proxy(method="OPTIONS", body=None)

    def do_HEAD(self):
        self._proxy(method="HEAD", body=None)

    def _proxy(self, method: str, body: bytes | None):
        url = BACKEND + self.path
        headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        if body is not None:
            headers["Content-Length"] = str(len(body))
        req = urllib.request.Request(url, data=body, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                chunked = resp.headers.get("Transfer-Encoding", "").lower() == "chunked"
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while True:
                    chunk = resp.read(65536)
                    if not chunk:
                        break
                    if chunked:
                        self.wfile.write(("%x\r\n" % len(chunk)).encode() + chunk + b"\r\n")
                    else:
                        self.wfile.write(chunk)
                if chunked:
                    self.wfile.write(b"0\r\n\r\n")
        except urllib.error.HTTPError as e:
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(e.read())
        except Exception as e:
            self.send_error(502, str(e))

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))


def main():
    server = HTTPServer(("0.0.0.0", PORT), ProxyHandler)
    print("chat_proxy: %s -> http://0.0.0.0:%s (tool_choice=required when tools present)" % (BACKEND, PORT), file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
