# Investigation: High RAM / grammar trigger bug vs vLLM

**Context:** "Grammar still awaiting trigger" + high RAM with GLM-4.7-Flash and tool definitions (llama.cpp #19068). This doc summarizes the investigation and comparison with vLLM as the gold standard. **No implementation** — investigation only.

**In this repo:** The fix is **already applied** in `external/llama.cpp`: for GLM 4.5, grammar is used only when `tool_choice == required`; for `tool_choice=auto` (server default), parse-only (no grammar) is used, so the RAM bug is avoided when clients use the default. See `external/llama.cpp/.pr-body.md` and `common/chat.cpp` (`common_chat_params_init_glm_4_5`).

---

## Root cause of high RAM (llama.cpp)

When grammar is used with a **lazy trigger** (e.g. `tool_choice=required` for GLM):

1. The grammar waits for a trigger string (e.g. `<tool_call>`) before constraining sampling.
2. Every token generated **before** the trigger is appended to:
   - `trigger_buffer` (string)
   - `trigger_buffer_positions` (vector of token + position)
3. If the model **never** outputs the trigger (bug #19068), these structures grow **unbounded** → high RAM and log spam ("Grammar still awaiting trigger after token N").

**Relevant code:**

- `external/llama.cpp/src/llama-grammar.cpp`: `llama_grammar_accept_impl` — when `grammar.awaiting_trigger` is true, each token is appended to `grammar.trigger_buffer` and `grammar.trigger_buffer_positions` (no cap).
- `external/llama.cpp/src/llama-grammar.h`: struct `llama_grammar` — `trigger_buffer`, `trigger_buffer_positions`.

---

## How vLLM does it (gold standard)

vLLM **does not** use a grammar trigger for tool calls:

- Tool calls are **parsed from decoded text** after (or while) the model generates.
- Regex / XML parsing on the output string: `<tool_call>...</tool_call>`, `<arg_key>...</arg_key><arg_value>...</arg_value>`.
- No "awaiting trigger" state → no unbounded buffer → no RAM growth from that path.

**Relevant code:**

- `external/vllm/vllm/tool_parsers/glm4_moe_tool_parser.py` — `Glm4MoeModelToolParser`: regex on `<tool_call>`, `<arg_key>`, `<arg_value>`; streaming extraction from text.
- `external/vllm/vllm/tool_parsers/glm47_moe_tool_parser.py` — same idea for GLM-4.7.
- `external/vllm/vllm/tool_parsers/abstract_tool_parser.py` — `extract_tool_calls` / `extract_tool_calls_streaming` on **model_output** (decoded text), not on a grammar trigger buffer.

For "required" / forced function, vLLM uses `StructuredOutputsParams(json=...)` (constrained decoding / JSON schema), not a lazy grammar with trigger buffer.

---

## What llama.cpp already does (GLM 4.5) — **fix is applied in this repo**

**This fix is already present** in this repo’s `external/llama.cpp` (from the fix/glm45-tool-parse-only-auto work; see `.pr-body.md` there). The runpod copy of llama.cpp includes it.

For **GLM 4.5** (`common_chat_params_init_glm_4_5` in `common/chat.cpp`):

- Grammar and `grammar_triggers` are **only** set when `tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED`.
- For `tool_choice == auto` (or none): **no grammar** — tool calls are detected by **parsing decoded text** (`common_chat_parse_glm_4_5`). Same idea as vLLM.

So:

- If the client sends `tool_choice=auto` (or omits it): server default is `"auto"` (`server-common.cpp`: `json_value(body, "tool_choice", std::string("auto"))`). Then llama.cpp uses the **parse-only** path for GLM 4.5 → no grammar trigger → **no RAM bug** (this is the path we fixed).
- If the client sends `tool_choice=required`: llama.cpp uses grammar + lazy trigger → if the model never outputs the trigger, `trigger_buffer` grows without bound → RAM bug can still occur (only in that case).

---

## When the bug happens

- Client sends `tool_choice=required` (or equivalent).
- llama.cpp uses grammar + lazy trigger for GLM.
- Model never outputs the trigger (e.g. stays in thinking or outputs something else).
- `trigger_buffer` and `trigger_buffer_positions` grow unbounded → high RAM.

---

## Possible mitigations (not implemented)

1. **Proxy/client:** Prefer `tool_choice=auto` when possible so llama.cpp uses parse-only (no grammar). Avoid sending `required` unless strictly needed.
2. **Upstream (llama.cpp):** Cap `trigger_buffer` size in `llama_grammar_accept_impl` (e.g. max 64KB or N tokens). When exceeded: clear buffer (sliding window or discard) and log once. Prevents unbounded RAM; trigger matching may fail if trigger was split across the discarded part.
3. **Upstream (llama.cpp):** For GLM, consider always using parse-only (no grammar) and forcing tool call by other means (e.g. prompt / sampling) when `tool_choice=required`, to align with vLLM and avoid the trigger path entirely.

---

## References

- Upstream: [llama.cpp #19068](https://github.com/ggml-org/llama.cpp/issues/19068)
- llama.cpp: `common/chat.cpp` (GLM 4.5 init), `src/llama-grammar.cpp`, `src/llama-grammar.h`
- vLLM: `vllm/tool_parsers/glm4_moe_tool_parser.py`, `glm47_moe_tool_parser.py`, `abstract_tool_parser.py`
- In this repo: `KNOWN_ISSUES.md` §3 (Grammar still awaiting trigger)
