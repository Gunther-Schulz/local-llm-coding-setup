#!/usr/bin/env python3
"""
HTTP Proxy Server for Tool Calls

Sits between Cursor IDE and Llama.cpp backend.
Intercepts tool calls, injects reminders, prevents loops via deduplication.

Usage:
    python server.py --port 8080 --backend-url http://localhost:8080 --config config/default_rules.yaml
"""

import argparse
import json
import logging
import os
import sys
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Dict, Any, Optional, List
from urllib.parse import urlparse, parse_qs
import urllib.request

from config.loader import load_rules, ConfigError
from interceptor import Interceptor


# Configure logging (level may be overridden after config load)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tool-proxy")
_interceptor_logger = logging.getLogger("interceptor")

# Truncate for logs (avoid huge payloads)
def _truncate(s: str, max_len: int = 1200) -> str:
    if not s or len(s) <= max_len:
        return s or ""
    return s[:max_len] + f" ... [truncated, total {len(s)} chars]"


def _content_preview(content: Any, max_len: int = 400) -> str:
    """String preview of message content (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return _truncate(content, max_len)
    if isinstance(content, list):
        parts = []
        for i, part in enumerate(content[:5]):
            if isinstance(part, dict):
                t = part.get("type", "?")
                text = part.get("text", part.get("content", str(part)))[:200]
                parts.append(f"{t}:{_truncate(str(text), 200)}")
            else:
                parts.append(_truncate(str(part), 150))
        return " | ".join(parts) + (f" ... [{len(content)} parts]" if len(content) > 5 else "")
    return _truncate(str(content), max_len)


class ToolProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler for tool proxy."""
    
    interceptor: Optional[Interceptor] = None
    backend_url: Optional[str] = None
    debug: bool = False
    _request_counter: int = 0  # used to synthesize turn_id when client sends none
    
    def log_message(self, format, *args):
        """Override to use our logging configuration."""
        logger.debug(f"{self.address_string()} - {format % args}")
    
    def _debug(self, msg: str, *args, **kwargs) -> None:
        """Log only when debug is enabled."""
        if getattr(self, "debug", False) or getattr(ToolProxyHandler, "debug", False):
            logger.debug(msg, *args, **kwargs)
    
    def _log_all_messages(self, request_data: Dict[str, Any]) -> None:
        """Log every message in the request for tracking."""
        if "messages" not in request_data:
            return
        messages: List[Dict] = request_data["messages"]
        logger.info("=== request messages (%d) ===", len(messages))
        for i, msg in enumerate(messages):
            role = msg.get("role", "?")
            content = msg.get("content", "")
            content_len = len(content) if isinstance(content, str) else (len(str(content)) if content is not None else 0)
            preview = _content_preview(content, 350)
            tool_calls = msg.get("tool_calls") or []
            if tool_calls:
                tc_summary = "; ".join(
                    f"id={tc.get('id', '?')} name={tc.get('function', {}).get('name', tc.get('name', '?'))}"
                    for tc in tool_calls[:10]
                )
                if len(tool_calls) > 10:
                    tc_summary += f" ... +{len(tool_calls) - 10} more"
                logger.info("  [msg %d] role=%s content_len=%d tool_calls=%d | %s", i, role, content_len, len(tool_calls), tc_summary)
                logger.info("  [msg %d] content_preview: %s", i, preview[:500] if preview else "(empty)")
            else:
                logger.info("  [msg %d] role=%s content_len=%d", i, role, content_len)
                if preview:
                    logger.info("  [msg %d] content_preview: %s", i, preview)
    
    def _log_all_tool_calls(self, tool_calls: List[Dict], source: str) -> None:
        """Log every tool call in full for tracking."""
        if not tool_calls:
            return
        logger.info("=== tool_calls (%d) source=%s ===", len(tool_calls), source)
        for i, tc in enumerate(tool_calls):
            fn = tc.get("function") or {}
            name = tc.get("tool") or tc.get("name") or (fn.get("name") if isinstance(fn, dict) else "?")
            call_id = tc.get("id", "?")
            params = tc.get("params") or tc.get("arguments")
            if params is None and isinstance(fn, dict):
                params = fn.get("arguments")
            if params is None:
                params = {}
            if isinstance(params, str):
                try:
                    params = json.loads(params) if params else {}
                except (json.JSONDecodeError, TypeError):
                    params = {"_raw": _truncate(params, 500)}
            if not isinstance(params, dict):
                params = {}
            params_str = json.dumps(params, indent=2)
            params_trunc = _truncate(params_str, 1500)
            logger.info("  [tc %d] id=%s name=%s", i, call_id, name)
            logger.info("  [tc %d] params: %s", i, params_trunc)
    
    def _log_response_content(self, response: Dict[str, Any]) -> None:
        """Log response message/choices for tracking."""
        if not isinstance(response, dict) or "error" in response:
            return
        choices = response.get("choices") or []
        if not choices:
            return
        logger.info("=== response choices (%d) ===", len(choices))
        for i, choice in enumerate(choices[:5]):
            msg = choice.get("message") or {}
            role = msg.get("role", "?")
            content = msg.get("content") or ""
            content_len = len(content) if isinstance(content, str) else 0
            tool_calls = msg.get("tool_calls") or []
            logger.info("  [choice %d] role=%s content_len=%d tool_calls=%d", i, role, content_len, len(tool_calls))
            if content:
                logger.info("  [choice %d] content_preview: %s", i, _truncate(content, 500))
            for j, tc in enumerate(tool_calls[:5]):
                fn = tc.get("function") or {}
                name = fn.get("name", "?")
                args_preview = _truncate(fn.get("arguments", ""), 400)
                logger.info("  [choice %d tc %d] name=%s arguments: %s", i, j, name, args_preview)
    
    def do_POST(self):
        """Handle POST requests (tool calls)."""
        t_start = time.perf_counter()
        client = self.address_string()
        try:
            parsed_path = urlparse(self.path)
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            
            logger.info("POST %s from %s Content-Length=%d", parsed_path.path, client, content_length)
            
            # Extract turn_id (or synthesize one per request so read coalescing works)
            turn_id = self._get_turn_id_from_headers()
            turn_id_source = "header/query"
            if turn_id is None and body:
                try:
                    request_data_pre = json.loads(body.decode('utf-8'))
                    turn_id = request_data_pre.get("turn_id")
                    turn_id_source = "body"
                except (json.JSONDecodeError, TypeError):
                    pass
            if turn_id is None:
                ToolProxyHandler._request_counter += 1
                turn_id = f"req-{ToolProxyHandler._request_counter}"
                turn_id_source = "synthetic"
            self._debug("turn_id=%s (from %s)", turn_id, turn_id_source)
            
            try:
                request_data = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError as e:
                logger.error("Invalid JSON from %s: %s", client, e)
                self._send_error(400, f"Invalid JSON: {e}")
                return
            
            req_keys = list(request_data.keys())
            self._debug("request keys: %s", req_keys)
            self._debug("request body: %s", _truncate(json.dumps(request_data, indent=2), 1600))
            
            # Log every message for tracking
            self._log_all_messages(request_data)
            
            tool_calls, extract_source = self._extract_tool_calls_with_source(request_data)
            self._debug("tool_calls extracted: count=%d source=%s", len(tool_calls) if tool_calls else 0, extract_source)
            
            # Log every tool call for tracking
            self._log_all_tool_calls(tool_calls, extract_source)
            
            reminders_collected: List[str] = []
            if tool_calls and self.interceptor:
                logger.info("intercepting %d tool call(s) turn_id=%s", len(tool_calls), turn_id)
                for i, tool_call in enumerate(tool_calls):
                    fn = tool_call.get("function") or {}
                    tool_name = (
                        tool_call.get("tool")
                        or tool_call.get("name")
                        or (fn.get("name") if isinstance(fn, dict) else "")
                        or ""
                    )
                    params = tool_call.get("params") or tool_call.get("arguments") or (fn.get("arguments") if isinstance(fn, dict) else {})
                    if isinstance(params, str):
                        try:
                            params = json.loads(params) if params else {}
                        except (json.JSONDecodeError, TypeError):
                            params = {}
                    param_keys = list(params.keys()) if isinstance(params, dict) else []
                    logger.info("  [%d] tool=%s params=%s", i, tool_name, param_keys)
                    self._debug("  [%d] full params: %s", i, _truncate(json.dumps(params, indent=2), 600))
                    
                    modified_call, reminder = self.interceptor.intercept_call(tool_call, turn_id)
                    
                    if reminder:
                        reminders_collected.append(reminder)
                        logger.info("  [%d] reminder for %s (len=%d)", i, tool_name, len(reminder))
                        self._debug("  [%d] reminder: %s", i, _truncate(reminder, 800))
            else:
                self._debug("no tool_calls to intercept")
            
            # Deliver reminders to the model via a synthetic user message (backend-safe)
            if reminders_collected:
                reminder_content = "[Tool reminders]\n\n" + "\n\n---\n\n".join(reminders_collected)
                if "messages" not in request_data:
                    request_data["messages"] = []
                request_data["messages"].append({
                    "role": "user",
                    "content": reminder_content,
                })
                self._debug("appended reminder message (%d parts, %d chars)", len(reminders_collected), len(reminder_content))
            
            # Strip any proxy-injected fields from tool_calls so backend receives valid schema
            self._strip_proxy_fields_from_request(request_data)
            t_before_backend = time.perf_counter()
            stream_requested = request_data.get("stream", False)
            backend_url_used = f"{self.backend_url or ''}{parsed_path.path}"
            self._debug("forwarding to backend: %s body_size=%d stream=%s", backend_url_used, len(json.dumps(request_data)), stream_requested)
            
            if stream_requested:
                self._forward_stream_to_backend(request_data, parsed_path.path)
                total_ms = (time.perf_counter() - t_start) * 1000
                logger.info("request done (streaming) in %.0f ms", total_ms)
                return
            
            response = self._forward_to_backend(request_data, parsed_path.path)
            
            t_after_backend = time.perf_counter()
            backend_ms = (t_after_backend - t_before_backend) * 1000
            resp_is_dict = isinstance(response, dict)
            resp_keys = list(response.keys()) if resp_is_dict else []
            resp_has_error = resp_is_dict and "error" in response
            
            logger.info("backend responded in %.0f ms keys=%s error=%s", backend_ms, resp_keys[:12], resp_has_error)
            if resp_has_error:
                logger.warning("backend error: %s", response.get("error") or response.get("details"))
                self._debug("backend response: %s", _truncate(json.dumps(response, indent=2), 2000))
                # Return 502 so clients see a real failure (e.g. Cursor "trouble connecting")
                err_msg = response.get("error") or response.get("details") or "Backend error"
                self._send_error(502, err_msg)
                return
            self._debug("backend response: %s", _truncate(json.dumps(response, indent=2), 2000))
            
            # Log response messages/choices for tracking
            self._log_response_content(response)
            
            self._send_response(response)
            
            t_done = time.perf_counter()
            total_ms = (t_done - t_start) * 1000
            logger.info("request done in %.0f ms (backend %.0f ms)", total_ms, backend_ms)
            self._debug("response body size=%d bytes", len(json.dumps(response)))
            
        except Exception as e:
            logger.error("request failed after %.0f ms: %s", (time.perf_counter() - t_start) * 1000, e, exc_info=True)
            self._send_error(500, f"Internal error: {e}")
    
    def do_GET(self):
        """Handle GET requests (health check, etc)."""
        try:
            parsed_path = urlparse(self.path)
            self._debug("GET %s from %s", parsed_path.path, self.address_string())
            if parsed_path.path == "/health" or parsed_path.path == "/":
                self._send_response({"status": "ok", "proxy": "tool-proxy"})
                logger.debug("GET %s -> 200 ok", parsed_path.path)
            else:
                logger.debug("GET %s -> 404", parsed_path.path)
                self._send_error(404, "Not found")
        except Exception as e:
            logger.error("GET request failed: %s", e, exc_info=True)
            self._send_error(500, f"Internal error: {e}")
    
    def _extract_tool_calls(self, request_data: Dict[str, Any]) -> list:
        """Extract tool calls from request data."""
        tool_calls, _ = self._extract_tool_calls_with_source(request_data)
        return tool_calls
    
    def _extract_tool_calls_with_source(self, request_data: Dict[str, Any]) -> tuple:
        """Extract tool calls and return (list, source_description)."""
        if "tool_calls" in request_data:
            return request_data["tool_calls"], "tool_calls"
        if "messages" in request_data:
            tool_calls = []
            for message in request_data["messages"]:
                if "tool_calls" in message:
                    tool_calls.extend(message["tool_calls"])
            return tool_calls, "messages[].tool_calls"
        if "tool" in request_data or "name" in request_data:
            return [request_data], "single_tool"
        return [], "none"
    
    def _get_turn_id_from_headers(self) -> Optional[str]:
        """Extract turn ID from request headers or query params (no body read)."""
        parsed = urlparse(self.path)
        query_params = parse_qs(parsed.query)
        if "turn_id" in query_params:
            return query_params["turn_id"][0]
        return self.headers.get("X-Turn-ID")
    
    def _strip_proxy_fields_from_request(self, request_data: Dict[str, Any]) -> None:
        """Remove proxy-injected fields from tool_calls so backend receives valid schema (mutates in place)."""
        for message in request_data.get("messages") or []:
            for tc in (message.get("tool_calls") or []):
                if isinstance(tc, dict) and "response_reminder" in tc:
                    del tc["response_reminder"]
    
    def _forward_to_backend(self, request_data: Dict[str, Any], path: str) -> Dict[str, Any]:
        """Forward request to backend Llama.cpp server."""
        if not self.backend_url:
            logger.warning("no backend_url configured")
            return {"error": "No backend URL configured"}
        
        backend_path = path.lstrip("/")
        backend_full_url = f"{self.backend_url.rstrip('/')}/{backend_path}"
        data = json.dumps(request_data).encode('utf-8')
        self._debug("backend request: url=%s body_bytes=%d", backend_full_url, len(data))
        
        req = urllib.request.Request(
            backend_full_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **{k: v for k, v in self.headers.items() if k not in ['Host', 'Content-Length']}
            },
            method="POST"
        )
        
        try:
            with urllib.request.urlopen(req, timeout=300) as response:
                body = response.read().decode('utf-8')
                self._debug("backend response: status=%s body_bytes=%d", getattr(response, 'status', 200), len(body))
                return json.loads(body)
        except urllib.error.HTTPError as e:
            try:
                body = e.fp.read().decode('utf-8') if e.fp else ""
            except Exception:
                body = ""
            logger.error("backend HTTP error: %s %s body_len=%d", e.code, e.reason, len(body))
            self._debug("backend error body: %s", _truncate(body, 500))
            return {"error": f"Backend error: {e.code}", "details": e.reason}
        except urllib.error.URLError as e:
            logger.error("backend URL error: %s", e.reason)
            return {"error": f"Backend unreachable: {e.reason}"}
        except Exception as e:
            logger.error("backend forward failed: %s", e, exc_info=True)
            return {"error": f"Backend error: {e}"}
    
    def _forward_stream_to_backend(self, request_data: Dict[str, Any], path: str) -> None:
        """Forward request to backend and stream response body to client (for stream=true)."""
        if not self.backend_url:
            self._send_error(502, "No backend URL configured")
            return
        backend_path = path.lstrip("/")
        backend_full_url = f"{self.backend_url.rstrip('/')}/{backend_path}"
        data = json.dumps(request_data).encode('utf-8')
        req = urllib.request.Request(
            backend_full_url,
            data=data,
            headers={
                "Content-Type": "application/json",
                **{k: v for k, v in self.headers.items() if k not in ['Host', 'Content-Length']}
            },
            method="POST"
        )
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                self.send_response(resp.status)
                # Copy headers relevant for streaming (avoid Content-Length; backend may use chunked)
                for name in ["Content-Type", "Transfer-Encoding", "Cache-Control", "X-Accel-Buffering"]:
                    if name in resp.headers:
                        self.send_header(name, resp.headers[name])
                self.end_headers()
                while True:
                    chunk = resp.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except urllib.error.HTTPError as e:
            try:
                body = e.fp.read().decode('utf-8') if e.fp else ""
            except Exception:
                body = ""
            logger.error("backend HTTP error (stream): %s %s", e.code, e.reason)
            self._send_error(502, f"Backend error: {e.code} {e.reason}")
        except urllib.error.URLError as e:
            logger.error("backend URL error (stream): %s", e.reason)
            self._send_error(502, f"Backend unreachable: {e.reason}")
        except Exception as e:
            logger.error("backend stream forward failed: %s", e, exc_info=True)
            self._send_error(502, f"Backend error: {e}")
    
    def _send_response(self, data: Dict[str, Any]) -> None:
        """Send JSON response to client."""
        body = json.dumps(data).encode('utf-8')
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)
    
    def _send_error(self, status_code: int, message: str) -> None:
        """Send error response."""
        body = json.dumps({"error": message}).encode('utf-8')
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)


class ToolProxyServer:
    """HTTP proxy server wrapper."""
    
    def __init__(self, port: int, backend_url: str, config_path: Optional[str] = None, verbose: bool = False):
        """
        Initialize the proxy server.
        
        Args:
            port: Port to listen on
            backend_url: URL of Llama.cpp backend
            config_path: Path to config file (optional)
            verbose: If True, enable debug logging (overrides config)
        """
        self.port = port
        self.backend_url = backend_url
        self.config_path = config_path
        
        # Load configuration
        try:
            self.config = load_rules(config_path)
            logger.info(f"Configuration loaded from {config_path or 'default'}")
        except ConfigError as e:
            logger.error(f"Failed to load config: {e}")
            sys.exit(1)
        
        # Initialize interceptor
        self.interceptor = Interceptor(self.config)
        
        # Apply logging config (level + debug)
        log_config = self.config.get("logging", {})
        level_name = (os.environ.get("LOG_LEVEL") or log_config.get("level") or "INFO").upper()
        level = getattr(logging, level_name, logging.INFO)
        debug = (
            verbose
            or os.environ.get("DEBUG", "").strip().lower() in ("1", "true", "yes")
            or log_config.get("debug", False)
        )
        if debug:
            level = logging.DEBUG
        logger.setLevel(level)
        _interceptor_logger.setLevel(level)
        if logging.getLogger().handlers:
            for h in logging.getLogger().handlers:
                h.setLevel(level)
        logger.info(f"Log level: {logging.getLevelName(level)} (debug={debug})")
        
        # Set up handler class
        ToolProxyHandler.interceptor = self.interceptor
        ToolProxyHandler.backend_url = backend_url
        ToolProxyHandler.debug = debug
        
        # Create server
        self.server = HTTPServer(("0.0.0.0", port), ToolProxyHandler)
        
        logger.info(f"Tool proxy server initialized on port {port}")
        logger.info(f"Forwarding to backend: {backend_url}")
    
    def run(self) -> None:
        """Run the server indefinitely."""
        logger.info(f"Starting tool proxy server on port {self.port}")
        logger.info("Press Ctrl+C to stop")
        
        try:
            self.server.serve_forever()
        except KeyboardInterrupt:
            logger.info("Shutting down...")
            self.server.shutdown()
    
    def stop(self) -> None:
        """Stop the server."""
        logger.info("Stopping server...")
        self.server.shutdown()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tool Call Proxy Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python server.py --port 8080 --backend-url http://localhost:8080
  python server.py --port 8080 --backend-url http://localhost:8080 --config config/default_rules.yaml
        """
    )
    
    parser.add_argument(
        "--port", "-p",
        type=int,
        default=8080,
        help="Port to listen on (default: 8080)"
    )
    
    parser.add_argument(
        "--backend-url", "-b",
        type=str,
        required=True,
        help="URL of Llama.cpp backend (e.g., http://localhost:8080)"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to config file (default: config/default_rules.yaml)"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Create and run server (--verbose enables debug logging)
    server = ToolProxyServer(
        port=args.port,
        backend_url=args.backend_url,
        config_path=args.config,
        verbose=args.verbose,
    )
    
    server.run()


if __name__ == "__main__":
    main()