# Compression vs 128k context (Cursor preprocessing)

From `logs/proxy.log`: **all Cursor traffic stayed well under 128k**. We still applied tool condense and sliding window, so we compressed more than necessary.

## What the log shows

- **context_limit**: 131072 (128k)
- **compression_threshold**: 100000, **trigger_messages**: 50
- **tokens_in**: typically ~20k–63k; **tokens_out** (after condense): ~19k–22k
- **tokens_in** never reached 100k; no request approached 128k

So every request would have **fit in 128k** without our condense/sliding window. We were shrinking content (tool preview 500 chars, summarization of old messages) even when the full conversation would have fit.

## Why we compressed anyway

1. **Tool condense** runs on every request: any tool response > 2000 chars → 500‑char preview. So we condense regardless of total prompt size.
2. **Sliding window** triggers when **messages > 50** OR **tokens > 100k**. So at 51 messages and 60k tokens we still summarized, even though 60k fits in 128k.

## Cursor preprocessing

Cursor does its own context/preprocessing on the client. What they send us is already bounded (e.g. by their own context or UI limits). So we are often **double-compressing**: Cursor sends a bounded request, and we condense/summarize again.

## Takeaway

With **128k context** and **Cursor as client**, it’s reasonable to:

- **Only condense tool responses when we’re near the limit** (e.g. only when `tokens_before_condense > 0.85 * context_limit`), so normal traffic gets full tool content.
- **Only trigger sliding window when tokens actually approach the limit** (e.g. raise `COMPRESSION_TRIGGER_MESSAGES` for 128k, or trigger only when `tokens > COMPRESSION_THRESHOLD`), so we don’t summarize just because message count hit 50.

That way we still protect against 413 when a request is genuinely huge, but we don’t compress when everything would have fit in 128k.

## Config: only compress when near limit

Set **COMPRESSION_ONLY_WHEN_NEAR_LIMIT=1** (and optionally **COMPRESSION_NEAR_LIMIT_FRACTION=0.85**) in `config/settings.env`:

- **Tool condense** runs only when `tokens_before_condense > context_limit * 0.85` (e.g. ~111k for 128k). Below that, full tool responses are passed through.
- **Sliding window** triggers only when `tokens > COMPRESSION_THRESHOLD` (message count is ignored). So we don’t summarize just because there are 50+ messages.

Restart the proxy after changing. See `stack/settings.py` for the defaults.
