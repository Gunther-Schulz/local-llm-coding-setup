#!/usr/bin/env python3
"""
Thin proxy in front of llama-server for coding.
- Forwards all requests to BACKEND_URL (e.g. http://127.0.0.1:8001).
- Per-model proxy features (default: all off). Enable in config/models/<key>.yaml:
  - proxy_force_tool_choice_required: true  -> when tools present, set tool_choice to "required"
  - proxy_loop_limits: true                 -> inject stop after N identical/similar tool calls
  Request "model" is matched to config key (filename without .yaml). Unknown model = all features off.
Env: BACKEND_URL, PROXY_PORT (8010), PROXY_DEBUG, PROXY_CONFIG_DIR (default: ROOT/config/models),
  PROXY_MAX_IDENTICAL_TOOL_CALLS, PROXY_MAX_SIMILAR_TOOL_CALLS.
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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_MODELS_DIR = os.environ.get("PROXY_CONFIG_DIR", os.path.join(ROOT, "config", "models"))

BACKEND = os.environ.get("BACKEND_URL", "http://127.0.0.1:8001").rstrip("/")
PORT = int(os.environ.get("PROXY_PORT", "8010"))
DEBUG = os.environ.get("PROXY_DEBUG", "").strip().lower() in ("1", "true", "on", "yes")
MAX_IDENTICAL_TOOL_CALLS = int(os.environ.get("PROXY_MAX_IDENTICAL_TOOL_CALLS", "3"))
MAX_SIMILAR_TOOL_CALLS = int(os.environ.get("PROXY_MAX_SIMILAR_TOOL_CALLS", "4"))

# Per-model proxy options: model_key -> {force_tool_choice_required: bool, loop_limits: bool}. Default both False.
PROXY_MODEL_CONFIG: dict[str, dict] = {}

def _load_proxy_config() -> dict[str, dict]:
    """Load per-model proxy options from config/models/*.yaml. Returns model_key -> {force_tool_choice_required, loop_limits}. Default for unknown model: both False."""
    result: dict[str, dict] = {}
    if not os.path.isdir(CONFIG_MODELS_DIR):
        return result
    try:
        import yaml
    except ImportError:
        return result
    for f in sorted(os.listdir(CONFIG_MODELS_DIR)):
        if not f.endswith(".yaml") or f.startswith("."):
            continue
        model_key = f[:-5]
        path = os.path.join(CONFIG_MODELS_DIR, f)
        try:
            with open(path, encoding="utf-8") as fp:
                data = yaml.safe_load(fp) or {}
            fc = data.get("proxy_force_tool_choice_required", False)
            ll = data.get("proxy_loop_limits", False)
            result[model_key] = {"force_tool_choice_required": bool(fc), "loop_limits": bool(ll)}
        except Exception:
            continue
    return result


# Redact auth for safe logging
def _redact_headers(headers: dict) -> dict:
    return {k: "***" if k.lower() in ("authorization", "api-key", "x-api-key") else v for k, v in headers.items()}


def _debug(msg: str, *args) -> None:
    if not DEBUG:
        return
    line = "[proxy][debug] " + (msg % args if args else msg)
    sys.stderr.write(line + "\n")
    sys.stderr.flush()


def _tool_call_signature(tc: dict) -> tuple[str, str]:
    """Return (name, normalized_args) for a tool call for comparison."""
    fn = tc.get("function") or {}
    name = (fn.get("name") or "").strip()
    args = fn.get("arguments") or ""
    try:
        args_obj = json.loads(args) if isinstance(args, str) else args
        args = json.dumps(args_obj, sort_keys=True)
    except Exception:
        pass
    return (name, args)


def _tool_call_similar_key(tc: dict) -> tuple[str, str] | None:
    """
    Return (name, key) for "similar" detection: same file (Read) or same pattern (Grep).
    Used to cap repetitive non-identical calls (e.g. Read config.py L1-30, L1-40, L1-50).
    """
    fn = tc.get("function") or {}
    name = (fn.get("name") or "").strip()
    args = fn.get("arguments") or ""
    try:
        args_obj = json.loads(args) if isinstance(args, str) else args
    except Exception:
        return None
    if not isinstance(args_obj, dict):
        return None
    if name == "Read":
        path = (args_obj.get("path") or args_obj.get("file_path") or "").strip()
        if path:
            return (name, path)
    if name == "Grep":
        pattern = (args_obj.get("pattern") or "").strip()
        if pattern:
            return (name, pattern)
    return None


def _apply_loop_limit(data: dict) -> dict:
    """
    C) Loop limit: if the same tool call (name+args) is repeated N times in a row, inject stop
    message and set tool_choice to "none" so the model answers instead of calling again.
    """
    messages = data.get("messages")
    if not messages or not isinstance(messages, list):
        return data
    tools = data.get("tools")
    if not tools or not isinstance(tools, list):
        return data

    # Assistant messages that contain tool_calls (one per "round")
    assistant_tool_rounds = []
    assistant_similar_rounds = []  # per round: set of (name, key) for Read path / Grep pattern
    for m in messages:
        if m.get("role") != "assistant":
            continue
        tcs = m.get("tool_calls")
        if not tcs or not isinstance(tcs, list):
            continue
        sigs = [_tool_call_signature(tc) for tc in tcs]
        assistant_tool_rounds.append(sigs)
        similar_keys = set()
        for tc in tcs:
            sk = _tool_call_similar_key(tc)
            if sk is not None:
                similar_keys.add(sk)
        assistant_similar_rounds.append(similar_keys)

    # 1) Count consecutive identical rounds (same name+args as the last one)
    identical_count = 0
    if assistant_tool_rounds:
        last_sigs = assistant_tool_rounds[-1]
        for i in range(len(assistant_tool_rounds) - 1, -1, -1):
            if assistant_tool_rounds[i] == last_sigs:
                identical_count += 1
            else:
                break
    if identical_count >= MAX_IDENTICAL_TOOL_CALLS:
        stop_msg = (
            "[System: The same tool call was repeated too many times (loop detected). Do NOT repeat that "
            "exact call again. In your reply: briefly summarize what you have found so far and what you "
            "will do next to continue the task. You may use other tools in the next turn.]"
        )
        data["messages"] = list(messages) + [{"role": "user", "content": stop_msg}]
        data["tool_choice"] = "none"
        if DEBUG:
            _debug(">>> LOOP LIMIT: identical_count=%d max=%d, injected stop (tool_choice=none)", identical_count, MAX_IDENTICAL_TOOL_CALLS)
        return data

    # 2) Count consecutive "similar" rounds: same file (Read) or same pattern (Grep)
    if assistant_similar_rounds and MAX_SIMILAR_TOOL_CALLS > 0:
        last_similar = assistant_similar_rounds[-1]
        for (name, key) in last_similar:
            similar_count = 0
            for i in range(len(assistant_similar_rounds) - 1, -1, -1):
                if (name, key) in assistant_similar_rounds[i]:
                    similar_count += 1
                else:
                    break
            if similar_count >= MAX_SIMILAR_TOOL_CALLS:
                if name == "Read":
                    stop_msg = (
                        "[System: You have already read this file multiple times. Do NOT read it again in small chunks. "
                        "Use the content you already have. In your reply: summarize what you found and what you will "
                        "do next. You may use other tools (e.g. different files or Grep) in the next turn.]"
                    )
                else:
                    stop_msg = (
                        "[System: You have already run this search/pattern multiple times. Do NOT repeat the same "
                        "search. Use the results you have. In your reply: summarize what you found and what you will "
                        "do next. You may use other tools in the next turn.]"
                    )
                data["messages"] = list(messages) + [{"role": "user", "content": stop_msg}]
                data["tool_choice"] = "none"
                if DEBUG:
                    _debug(">>> SIMILAR LIMIT: %s key=%r count=%d max=%d, injected stop (tool_choice=none)", name, key, similar_count, MAX_SIMILAR_TOOL_CALLS)
                return data

    return data


def apply_tool_choice(body: bytes) -> bytes:
    try:
        data = json.loads(body)
    except json.JSONDecodeError:
        return body
    tools = data.get("tools")
    if not tools or not isinstance(tools, list) or len(tools) == 0:
        return body
    model = (data.get("model") or "").strip() or None
    opts = PROXY_MODEL_CONFIG.get(model or "", {})
    force_required = opts.get("force_tool_choice_required", False)
    loop_limits = opts.get("loop_limits", False)

    if force_required:
        choice = data.get("tool_choice")
        if choice != "required" and choice != "none":
            data["tool_choice"] = "required"
    if loop_limits:
        data = _apply_loop_limit(data)
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
                    if k.lower() not in ("transfer-encoding", "connection", "content-length"):
                        self.send_header(k, v)
                # Do NOT forward Transfer-Encoding: chunked — Cursor expects raw SSE (data: ...\n\n).
                # Stream raw body bytes so the client sees "data: " from the first byte.
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
                        try:
                            self.wfile.write(chunk)
                        except (BrokenPipeError, ConnectionResetError):
                            if DEBUG:
                                _debug("<<< STREAM: client disconnected after %d chunks", chunk_num)
                            return
                    if DEBUG:
                        _debug("<<< STREAM END: %d chunks, %d bytes total", chunk_num, total)
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
        except (BrokenPipeError, ConnectionResetError) as e:
            if DEBUG:
                _debug("<<< CLIENT GONE: %s", e)
            return
        except Exception as e:
            if DEBUG:
                _debug("<<< EXCEPTION: %s", e)
            try:
                self.send_error(502, str(e))
            except (BrokenPipeError, ConnectionResetError):
                return

    def log_message(self, format, *args):
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))


def main():
    global DEBUG, PROXY_MODEL_CONFIG
    p = argparse.ArgumentParser(description="Chat proxy: forward to llama-server; per-model features from config/models/*.yaml (default: off).")
    p.add_argument("--debug", action="store_true", help="Log full requests and responses (env: PROXY_DEBUG=1)")
    args = p.parse_args()
    if args.debug:
        DEBUG = True
    PROXY_MODEL_CONFIG.update(_load_proxy_config())
    enabled = [k for k, v in PROXY_MODEL_CONFIG.items() if v.get("force_tool_choice_required") or v.get("loop_limits")]
    server = HTTPServer(("0.0.0.0", PORT), ProxyHandler)
    opts = []
    if DEBUG:
        opts.append("debug ON (request/response logging)")
    else:
        opts.append("debug OFF — use --debug or PROXY_DEBUG=1 for request/response output")
    opts.append("max_identical=%d" % MAX_IDENTICAL_TOOL_CALLS)
    opts.append("max_similar=%d" % MAX_SIMILAR_TOOL_CALLS)
    if enabled:
        opts.append("features ON for: " + ", ".join(enabled))
    else:
        opts.append("per-model features: all off (set proxy_* in config/models/<key>.yaml to enable)")
    opt_str = " [" + ", ".join(opts) + "]"
    print("chat_proxy: %s -> http://0.0.0.0:%s%s" % (BACKEND, PORT, opt_str), file=sys.stderr)
    server.serve_forever()


if __name__ == "__main__":
    main()
