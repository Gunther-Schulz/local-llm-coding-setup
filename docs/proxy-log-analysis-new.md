# Proxy log analysis (new log – post mitigations)

Analysis of `logs/proxy.log` (83,217 lines) after no-condense patterns, first-user preservation, and optional near-limit mode were added.

---

## 1. Config in use

- **context_limit**: 131,072 (128k)
- **compression_threshold**: 100,000
- **trigger_messages**: 50
- **window_size**: 40
- **COMPRESSION_ONLY_WHEN_NEAR_LIMIT**: not set (default 0) — normal condense/sliding on every request / at 50 messages

---

## 2. No-condense for instruction/proxy files: working

- **243** log lines: `No condense for tool result (path matches): ...` (CLIPPY_MKII.md, proxy/vision_router.py, etc.).
- **5** requests had **tool_condense: 0** (early in session when only CLIPPY/proxy reads were in the request).
- Example from end of log: `No condense for tool result (path matches): /home/g/dev/local/runpod/CLIPPY_MKII.md` and same for `proxy/vision_router.py`.

So **CLIPPY and proxy/*.py** (and other matched paths) are getting **full content**; only other large tool responses are condensed.

---

## 3. Token and message progression

| Phase        | messages_in | tokens_in (sample) | Notes                          |
|-------------|-------------|--------------------|--------------------------------|
| Start       | 3–29        | 20k → 66k          | tool_condense 0 then 1–6       |
| Mid         | 31–63       | 68k → 92k          | sliding_window still no        |
| New conv    | 3–49        | 17k → 69k          | second conversation segment    |
| 50+ msgs    | 51–65       | 69k → 89k          | sliding_window starts (trigger: messages>50) |
| Large       | 67–83       | **112k → 116,770**  | tokens_in exceeds 100k; condense + sliding |

- **Peak tokens_in**: **116,770** (83 messages) — would be close to 128k without condense.
- **tokens_out** after condense + sliding for that request: 71,842 then **41,189** (summarization + 40-message window).

---

## 4. Sliding window

- **Sliding window fired** when **messages > 50** (and sometimes tokens > 100k).
- At **83 messages, 116,770 tokens_in**: trigger `messages>50`, 42 old messages summarized (3.71s), 40 recent kept, **tokens_out 41,189**.
- **First user message** is preserved in the summary block (`[Initial user request]:` + content); the log shows “Context management: 42 old, 40 recent” and “Final context: 42 messages (1 summary + 40 recent)”.

So **first-user preservation** is in effect when the sliding window runs.

---

## 5. Tool condense (non–no-condense)

- For requests that *do* condense: **tool_condense** goes from 1 up to **19** condensed responses (e.g. at 83 messages).
- Those are large tool results that are **not** CLIPPY/proxy (e.g. other Read results, Grep, etc.); they get the 500-char preview.

---

## 6. Comparison with previous (pre-mitigation) log

| Aspect              | Previous log           | New log                          |
|---------------------|------------------------|----------------------------------|
| CLIPPY / proxy      | Condensed to 500 chars | **Full content** (no-condense)   |
| First user in summary | Not preserved       | **Preserved** ([Initial user request]) |
| Sliding trigger     | messages>50 or tokens>100k | Same (messages>50 fired)    |
| Peak tokens_in     | ~63k observed          | **116,770** (one request)        |
| COMPRESSION_ONLY_WHEN_NEAR_LIMIT | N/A           | **Off** (not enabled in this run) |

---

## 7. Conclusions

1. **No-condense patterns** are doing what we want: CLIPPY_MKII.md and proxy/vision_router.py (and matched paths) are not condensed; the model gets full file content for those.
2. **First user message** is kept in the summary when the sliding window runs, so the original task (“use CLIPPY…”) stays in context.
3. **Sliding window** still triggers on **message count** (50+) in this log because **COMPRESSION_ONLY_WHEN_NEAR_LIMIT** was not set. So we summarized at 51 messages even when tokens were ~81k (under 100k and under 128k).
4. **One request reached 116,770 tokens_in** — without tool condense that could have been near or over 128k; condense + sliding brought it down to 41,189 tokens_out.
5. **If you enable COMPRESSION_ONLY_WHEN_NEAR_LIMIT=1**: we would only condense when tokens > ~111k (0.85×128k) and would only trigger sliding when tokens > 100k (no message-count trigger). That would reduce summarization for conversations that stay under 100k (e.g. 51 messages at 81k tokens).

---

## 8. Recommendation

For Cursor + 128k where most traffic stays under 100k: set **COMPRESSION_ONLY_WHEN_NEAR_LIMIT=1** in `config/settings.env` so we only condense and slide when the request is actually near the context limit. The new log shows that with the current settings we still compress (and summarize) once message count passes 50, even when tokens would have fit in 128k.
