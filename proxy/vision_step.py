"""Two-step Code+Vision: get image description from vision (no tools), then send to coding with description."""
from __future__ import annotations

import json
import socket
import sys
from urllib.parse import urlparse

from .forward import log

VISION_PROMPT = (
    "Describe this image in detail for a coding assistant. Be concise but complete. "
    "Do not use tools; output only the description."
)


def _find_last_user_message_with_image(data: dict) -> tuple[list, str, int] | None:
    """
    Find the last user message that contains an image.
    Returns (content_parts_list, extracted_text, index) or None.
    """
    messages = data.get("messages")
    if not isinstance(messages, list):
        return None
    text_parts: list[str] = []
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            continue
        has_image = False
        for part in content:
            if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                has_image = True
                break
        if not has_image:
            continue
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text":
                    text_parts.append(part.get("text") or "")
                elif part.get("type") in ("image_url", "image"):
                    pass  # keep in content for vision
        user_text = " ".join(text_parts).strip() if text_parts else "Describe what you see."
        return (content, user_text, i)
    return None


def _build_vision_request_body(data: dict, content: list, user_text: str) -> bytes:
    """Build a minimal chat/completions request for vision: one user message, no tools."""
    prompt = user_text if user_text and user_text != "Describe what you see." else VISION_PROMPT
    # Ensure we have text + image so the model knows what to do
    parts = [{"type": "text", "text": prompt}]
    for part in content:
        if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
            parts.append(part)
    vision_messages = [{"role": "user", "content": parts}]
    vision_body = {
        "model": data.get("model", "local"),
        "messages": vision_messages,
        "stream": False,
        "max_tokens": 1024,
    }
    return json.dumps(vision_body).encode("utf-8")


def get_vision_description(
    vision_url: str,
    body: bytes,
    req_headers: dict,
    *,
    debug: bool = False,
) -> tuple[str, str]:
    """
    Call vision backend with a simplified request (image + describe prompt, no tools).
    Returns (description_text, user_text_from_message).
    """
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return ("", "")

    found = _find_last_user_message_with_image(data)
    if not found:
        return ("", "")

    content, user_text, _ = found
    vision_body_bytes = _build_vision_request_body(data, content, user_text)

    parsed = urlparse(vision_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    path = "/v1/chat/completions"

    headers = dict(req_headers)
    headers["Host"] = f"{host}:{port}" if port not in (80, 443) else host
    headers["Content-Length"] = str(len(vision_body_bytes))
    headers.pop("Connection", None)
    # Request non-streaming response so we can read the full content
    headers["Accept"] = "application/json"

    req_lines = [f"POST {path} HTTP/1.1"]
    for k, v in headers.items():
        req_lines.append(f"{k}: {v}")
    req_bytes = "\r\n".join(req_lines).encode("latin-1") + b"\r\n\r\n" + vision_body_bytes

    if debug:
        log("vision_step: POST %s (stream=false) -> %s", path, vision_url)

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(120)
    try:
        sock.connect((host, port))
        sock.sendall(req_bytes)

        buf = b""
        while b"\r\n\r\n" not in buf and b"\n\n" not in buf:
            chunk = sock.recv(4096)
            if not chunk:
                break
            buf += chunk
        if not buf:
            return ("", user_text)

        sep = b"\r\n\r\n" if b"\r\n\r\n" in buf else b"\n\n"
        head_part, _, body_start = buf.partition(sep)
        # Content-Length for body (we may have read some body already)
        resp_headers = {}
        for line in head_part.split(b"\r\n" if b"\r\n" in head_part else b"\n")[1:]:
            if b":" in line:
                k, _, v = line.partition(b":")
                resp_headers[k.strip().decode("latin-1").lower()] = v.strip().decode("latin-1")
        cl = int(resp_headers.get("content-length", 0))
        body_buf = body_start
        while len(body_buf) < cl:
            chunk = sock.recv(8192)
            if not chunk:
                break
            body_buf += chunk
        sock.close()

        resp_data = json.loads(body_buf.decode("utf-8", errors="replace"))
        choices = resp_data.get("choices") or []
        if not choices:
            return ("", user_text)
        msg = choices[0].get("message") or {}
        desc = (msg.get("content") or "").strip()
        return (desc, user_text)
    except (socket.timeout, OSError, ConnectionRefusedError, json.JSONDecodeError, KeyError) as e:
        if debug:
            log("vision_step: %s", e)
        return ("", user_text)
    finally:
        try:
            sock.close()
        except Exception:
            pass


def build_coding_body_with_description(
    original_body: bytes,
    description: str,
    user_text: str,
) -> bytes:
    """
    Replace the user message that contained the image with a text message that includes
    the vision description. Tools and system message unchanged; coding model sees text only.
    """
    try:
        data = json.loads(original_body)
    except (json.JSONDecodeError, TypeError):
        return original_body

    found = _find_last_user_message_with_image(data)
    if not found:
        return original_body

    _, _, idx = found
    messages = list(data.get("messages") or [])

    if description:
        new_content = (
            f"The user attached an image and asked: {user_text}\n\n"
            f"Image description: {description}\n\n"
            "Please respond as the coding assistant."
        )
    else:
        new_content = (
            f"The user attached an image and asked: {user_text}\n\n"
            "(Image description unavailable.) Please respond as the coding assistant."
        )

    messages[idx] = {"role": "user", "content": new_content}
    data["messages"] = messages
    # Ensure we stream the coding response back to the client
    data["stream"] = True
    return json.dumps(data).encode("utf-8")
