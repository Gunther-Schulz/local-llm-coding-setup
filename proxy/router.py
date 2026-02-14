"""Route requests to the appropriate backend (e.g. image-containing chat → vision, else → coding)."""
from __future__ import annotations

import json
from .config import ProxyConfig


def request_has_image(body: bytes) -> bool:
    """
    Return True if the request body is a JSON chat/completions payload that contains an image.
    OpenAI/Cursor format: messages[].content can be a list of parts with type "image_url" or "image".
    """
    if not body or not body.strip():
        return False
    try:
        data = json.loads(body)
    except (json.JSONDecodeError, TypeError):
        return False

    messages = data.get("messages")
    if not isinstance(messages, list):
        return False

    for msg in messages:
        content = msg.get("content")
        if content is None:
            continue
        if isinstance(content, str):
            continue
        if isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") in ("image_url", "image"):
                    return True
    return False


def get_backend_url(method: str, path: str, body: bytes | None, config: ProxyConfig) -> str:
    """
    Choose backend URL for this request. In code_vision mode, POST /v1/chat/completions
    with an image in the body goes to the vision server; everything else goes to the coding server.
    """
    if not config.is_code_vision or not config.vision_url or not config.coding_url:
        return config.backend_url

    if method != "POST" or not path.rstrip("/").endswith("/v1/chat/completions"):
        return config.coding_url

    if body and request_has_image(body):
        return config.vision_url

    return config.coding_url
