#!/usr/bin/env python3
"""
Thin proxy in front of llama-server for coding (Qwen3 Coder).
- Forwards all requests to BACKEND_URL (e.g. http://127.0.0.1:8001).
- For POST /v1/chat/completions: if body has "tools" and tool_choice is missing or "auto",
  sets tool_choice to "required" before forwarding (avoids client-side control).
Env: BACKEND_URL, PROXY_PORT (default 8010), PROXY_CONFIG (YAML path, e.g. config/proxy.yaml), PROXY_DEBUG (1 = verbose logs).
YAML: parallel_tool_calls: true -> set in request when tools present (llama.cpp optional).
Usage: ./start-proxy.sh [--debug]  or  BACKEND_URL=... PROXY_PORT=8010 ./scripts/chat_proxy.py [--debug]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler

try:
    import yaml
except ImportError:
    yaml = None

BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")
PORT = int(os.environ.get("PROXY_PORT", "8010"))
DEBUG = os.environ.get("PROXY_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")

# Proxy options from config/proxy.yaml (parallel_tool_calls, etc.)
def _load_proxy_config() -> dict:
    path = os.environ.get("PROXY_CONFIG")
    if not path and os.path.isfile("config/proxy.yaml"):
        path = "config/proxy.yaml"
    if not path or not os.path.isfile(path):
        return {}
    if not yaml:
        return {}
    try:
        with open(path) as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}

PROXY_CONFIG = _load_proxy_config()

# Redact auth for safe logging
def _redact_headers(headers: dict) -> dict:
    return {k: "***" if k.lower() in ("authorization", "api-key", "x-api-key") else v for k, v in headers.items()}


def _debug(msg: str, *args) -> None:
    if not DEBUG:
        return
    line = "[proxy][debug] " + (msg % args if args else msg)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


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
    # Optional: enable parallel tool calls from config/proxy.yaml
    if PROXY_CONFIG.get("parallel_tool_calls") is True:
        data["parallel_tool_calls"] = True
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
        req_headers = {k: v for k, v in self.headers.items() if k.lower() not in ("host", "connection")}
        if body is not None:
            req_headers["Content-Length"] = str(len(body))

        if DEBUG:
            _debug(">>> REQUEST %s %s", method, self.path)
            _debug(">>> REQUEST HEADERS: %s", json.dumps(_redact_headers(dict(self.headers)), sort_keys=True))
            if body:
                try:
                    req_body_str = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
                    _debug(">>> REQUEST BODY:\n%s%s", req_body_str[:20000], "..." if len(req_body_str) > 20000 else "")
                    if len(req_body_str) > 20000:
                        _debug(">>> REQUEST BODY total length: %d chars (truncated above)", len(req_body_str))
                except Exception:
                    _debug(">>> REQUEST BODY (raw, first 2000 chars): %s", body[:2000].decode("utf-8", errors="replace"))

        req = urllib.request.Request(url, data=body, headers=req_headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                chunked = resp.headers.get("Transfer-Encoding", "").lower() == "chunked"
                resp_headers = dict(resp.headers)

                if DEBUG:
                    _debug("<<< RESPONSE %s", resp.status)
                    _debug("<<< RESPONSE HEADERS: %s", json.dumps(_redact_headers(resp_headers), sort_keys=True))

                self.send_response(resp.status)
                for k, v in resp.headers.items():
                    if k.lower() not in ("transfer-encoding", "connection"):
                        self.send_header(k, v)
                if chunked:
                    self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()

                if chunked:
                    total = 0
                    chunk_num = 0
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        chunk_num += 1
                        total += len(chunk)
                        if DEBUG and chunk_num <= 3:
                            preview = chunk[:500].decode("utf-8", errors="replace").replace("\r", " ").replace("\n", " ")
                            _debug("<<< STREAM CHUNK #%d %d bytes: %s%s", chunk_num, len(chunk), preview, "..." if len(chunk) > 500 else "")
                        self.wfile.write(("%x\r\n" % len(chunk)).encode() + chunk + b"\r\n")
                    if DEBUG:
                        _debug("<<< STREAM END: %d chunks, %d bytes total", chunk_num, total)
                    self.wfile.write(b"0\r\n\r\n")
                else:
                    raw = resp.read()
                    if DEBUG:
                        try:
                            _debug("<<< RESPONSE BODY:\n%s", json.dumps(json.loads(raw), indent=2, ensure_ascii=False)[:10000])
                        except Exception:
                            _debug("<<< RESPONSE BODY (raw, first 2000 chars): %s", raw[:2000].decode("utf-8", errors="replace"))
                        if len(raw) > 2000:
                            _debug("<<< RESPONSE BODY total length: %d bytes", len(raw))
                    self.wfile.write(raw)
        except urllib.error.HTTPError as e:
            err_body = e.read()
            if DEBUG:
                _debug("<<< ERROR RESPONSE %s", e.code)
                _debug("<<< ERROR HEADERS: %s", json.dumps(_redact_headers(dict(e.headers)), sort_keys=True))
                _debug("<<< ERROR BODY: %s", err_body[:2000].decode("utf-8", errors="replace"))
            self.send_response(e.code)
            for k, v in e.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(err_body)
        except Exception as e:
            if DEBUG:
                _debug("<<< EXCEPTION: %s", e)
            self.send_error(502, str(e))

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))


def main():
    global DEBUG
    p = argparse.ArgumentParser(description="Chat proxy: forward to llama-server, force tool_choice=required when tools present.")
    p.add_argument("--debug", action="store_true", help="Log full requests and responses (env: PROXY_DEBUG=1)")
    args = p.parse_args()
    if args.debug:
        DEBUG = True
    server = HTTPServer(("0.0.0.0", PORT), ProxyHandler)
    opts = []
    if DEBUG:
        opts.append("debug ON")
    if PROXY_CONFIG.get("parallel_tool_calls") is True:
        opts.append("parallel_tool_calls=true")
    opt_str = " [" + ", ".join(opts) + "]" if opts else ""
    print("chat_proxy: %s -> http://0.0.0.0:%s (tool_choice=required when tools present)%s" % (BACKEND, PORT, opt_str), file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
