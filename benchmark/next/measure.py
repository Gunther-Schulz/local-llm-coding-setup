#!/usr/bin/env python3
"""
Call llama-server /v1/chat/completions and report prompt tokens, completion tokens, time, tok/s.
Usage: measure.py [--port PORT] [--prompt-file FILE] [--max-tokens N]
  Default: port 18999, short built-in prompt, max_tokens 128.
  With --prompt-file: use file content as user message (long-context test).
"""
import argparse
import sys
import time

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=18999)
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--model", default="qwen3-coder-next")
    args = ap.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    if args.prompt_file:
        with open(args.prompt_file) as f:
            content = f.read()
    else:
        content = "Write a short Python function that returns the sum of two numbers. No explanation."

    try:
        import urllib.request
        import json
    except ImportError:
        try:
            import requests
        except ImportError:
            print("Need urllib or requests", file=sys.stderr)
            sys.exit(1)
        def post(url, data):
            r = requests.post(url, json=data, timeout=300)
            return r.json()
    else:
        def post(url, data):
            req = urllib.request.Request(url, data=json.dumps(data).encode(), method="POST",
                headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read().decode())

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "stream": False,
    }

    t0 = time.perf_counter()
    try:
        out = post(f"{base}/v1/chat/completions", payload)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    t1 = time.perf_counter()
    elapsed = t1 - t0

    usage = out.get("usage", {})
    prompt_tok = usage.get("prompt_tokens") or 0
    compl_tok = usage.get("completion_tokens") or 0
    if compl_tok > 0 and elapsed > 0:
        tok_s = compl_tok / elapsed
    else:
        tok_s = 0.0

    print(f"prompt_tokens={prompt_tok} completion_tokens={compl_tok} time={elapsed:.2f}s tok/s={tok_s:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
