#!/usr/bin/env python3
"""
Call llama-server /v1/chat/completions and report prompt tokens, completion tokens, time, tok/s.
With --prompt-file (long context): uses streaming to report gen_tok_s (generation/decode speed) for comparison.
Usage: measure.py [--port PORT] [--prompt-file FILE] [--max-tokens N]
  Default: port 18999, short built-in prompt, max_tokens 128.
"""
import argparse
import json
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

    import urllib.request

    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
    }

    # Long context: use streaming to measure generation speed (decode tok/s) separately from prefill
    if args.prompt_file:
        payload["stream"] = True
        payload["stream_options"] = {"include_usage": True}  # llama-server sends usage in last chunk only when this is set
        req = urllib.request.Request(
            f"{base}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        t0 = time.perf_counter()
        t_first = None
        t_last = None
        prompt_tok = 0
        compl_tok = 0
        compl_tok_counted = 0  # fallback: count content deltas when usage is missing
        try:
            with urllib.request.urlopen(req, timeout=600) as r:
                for line in r:
                    line = line.decode("utf-8").strip()
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        chunk = json.loads(line[6:])
                    except json.JSONDecodeError:
                        continue
                    t_now = time.perf_counter()
                    usage = chunk.get("usage", {})
                    if usage:
                        prompt_tok = usage.get("prompt_tokens") or prompt_tok
                        compl_tok = usage.get("completion_tokens") or compl_tok
                    delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
                    if delta.get("content") is not None:
                        compl_tok_counted += 1
                        if t_first is None:
                            t_first = t_now
                        t_last = t_now
            if compl_tok <= 0 and compl_tok_counted > 0:
                compl_tok = compl_tok_counted
            total_time = time.perf_counter() - t0
            decode_time = (t_last - t_first) if (t_first is not None and t_last is not None) else 0
            gen_tok_s = (compl_tok / decode_time) if (compl_tok > 0 and decode_time > 0) else 0.0
            print(f"prompt_tokens={prompt_tok} completion_tokens={compl_tok} time={total_time:.2f}s tok/s={compl_tok/total_time if total_time else 0:.2f} gen_tok_s={gen_tok_s:.2f}")
        except Exception as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)
        return 0

    # Short: non-streaming, tok/s is effectively generation speed
    payload["stream"] = False
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            out = json.loads(r.read().decode())
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    t1 = time.perf_counter()
    elapsed = t1 - t0
    usage = out.get("usage", {})
    prompt_tok = usage.get("prompt_tokens") or 0
    compl_tok = usage.get("completion_tokens") or 0
    tok_s = (compl_tok / elapsed) if (compl_tok > 0 and elapsed > 0) else 0.0
    print(f"prompt_tokens={prompt_tok} completion_tokens={compl_tok} time={elapsed:.2f}s tok/s={tok_s:.2f}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
