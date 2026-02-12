#!/usr/bin/env python3
"""
Chat client for the local llama-server (OpenAI-compatible /v1/chat/completions).
Uses the OpenAI Python library for API calls and Rich for Markdown-rendered replies.

Requires: pip install -r scripts/requirements-chat.txt
Usage:
  python scripts/chat.py                    # interactive, URL from config
  python scripts/chat.py "Your message"     # one-shot
  python scripts/chat.py http://host:port   # interactive, custom URL
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown

# Load config: repo root = parent of scripts/
ROOT = Path(__file__).resolve().parent.parent
SERVER_ENV = ROOT / "config" / "server.env"

def load_server_env() -> None:
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

def get_base_url() -> str:
    host = os.environ.get("HOST", "127.0.0.1")
    port = os.environ.get("PORT", "8000")
    return f"http://{host}:{port}/v1"

def main() -> None:
    load_server_env()

    # CLI: optional custom URL (first arg if it looks like a URL)
    args = sys.argv[1:]
    if args and re.match(r"^https?://", args[0]):
        base_url = args[0].rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        args = args[1:]
    else:
        base_url = get_base_url()

    one_shot = " ".join(args).strip() if args else None

    client = OpenAI(base_url=base_url, api_key="dummy")
    console = Console()
    url = base_url.replace("/v1", "")
    console.print(f"Chat with llama-server at [dim]{url}[/dim] (Ctrl+C to exit)\n")

    messages: list[dict[str, str]] = []

    def chat_turn(user_content: str) -> None:
        nonlocal messages
        messages.append({"role": "user", "content": user_content})

        reply_parts: list[str] = []
        stream = client.chat.completions.create(
            model="llama",
            messages=messages,
            stream=True,
            max_tokens=1024,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta if chunk.choices else None
            if delta and getattr(delta, "content", None):
                reply_parts.append(delta.content)
                console.print(delta.content, end="")

        full_reply = "".join(reply_parts)
        if full_reply.strip():
            console.print()  # newline after stream
            console.print(Markdown(full_reply))
        console.print()
        messages.append({"role": "assistant", "content": full_reply})

    if one_shot:
        chat_turn(one_shot)
        return

    while True:
        try:
            line = console.input("[bold green]> [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print()
            break
        if not line:
            continue
        chat_turn(line)

if __name__ == "__main__":
    main()
