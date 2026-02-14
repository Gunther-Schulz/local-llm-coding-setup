# Known issues: GLM + llama.cpp

This document describes problems we see when running **GLM-4.x** (especially GLM-4.7-Flash) GGUF with **llama-server** (llama.cpp). Our GLM model configs in `config/models/` reference this file; mitigations are applied there where possible.

---

## 1. repeat_penalty must be disabled

**Symptom:** Poor or unstable output when using default repeat penalty with GLM-4.7-Flash.

**Cause:** GLM-4.7-Flash behaves better with repeat penalty turned off.

**Fix:** In the model YAML set `repeat_penalty: 1.0` (effectively disables it). All our GLM configs already set this.

---

## 2. Do not use --cache-reuse with GLM

**Symptom:** Crashes, corruption, or bad behavior when reusing KV cache across requests.

**Cause:** llama.cpp’s `--cache-reuse` is not reliable with GLM models.

**Fix:** Do **not** pass `--cache-reuse` when serving GLM. Our `run_server.sh` does not add it; if you run llama-server manually or via another script, omit this flag. Our GLM YAMLs document this in a comment.

---

## 3. "Grammar still awaiting trigger" and tool-call corruption (llama.cpp #19068)

**Symptom:** Logs repeatedly show:

```text
Grammar still awaiting trigger after token N (`piece`)
```

- **Benign case:** The model is still in preamble, thinking, or plain text. Once it outputs `<tool_call>`, the tool-call grammar activates and parsing works. Occasional messages are normal.
- **Bug case:** The message repeats for thousands of tokens, output becomes gibberish, and RAM usage grows. The server has entered a bad state. Reported upstream as [llama.cpp issue #19068](https://github.com/ggml-org/llama.cpp/issues/19068). Seen with GLM-4.7-Flash when using tool definitions (OpenAI format via `--jinja`); the grammar trigger is not reliably applied.

**Mitigations:**

1. **With proxy:** Set `proxy_force_tool_choice_required: true` in the model YAML so the chat proxy sends `tool_choice: "required"`. That makes the grammar active from the start and reduces the chance of the trigger never firing. Our GLM configs use this.
2. **Without proxy:** If your client (e.g. Cursor) supports it, send `tool_choice: "required"` when using tools with this model.
3. Avoid malformed prompts (e.g. consecutive user messages without an assistant turn).
4. If you see sustained “awaiting trigger” plus gibberish and high RAM: force-kill and restart the server.
5. Run without `--verbose` to cut down log noise from this message.

See also **config/templates/README.md** (section “Grammar still awaiting trigger”) and **config/templates/GLM-4-tool.jinja** for the template that instructs the model to respond with a tool call first when tools are provided.

---

## 4. High CPU when model is partly offloaded

**Symptom:** Very high CPU usage when the GLM model does not fully fit in VRAM and some layers run on CPU.

**Cause:** Partially offloaded GLM with default server options can stress the CPU.

**Mitigation (optional):** In the model YAML you can try:

- `flash_attn: off`
- `cache_type_k: bf16`
- `cache_type_v: bf16`

Our GLM configs have these commented out; uncomment to experiment. They may reduce CPU load at the cost of some performance or compatibility. If the model fits fully in VRAM, you usually don’t need these.

---

## 5. "Does not natively describe tools" warning

**Symptom:** Server logs warn that the chat template supports tool calls but does not natively describe tools (so a generic fallback is used).

**Cause:** The default or embedded GLM template doesn’t inject tool definitions in the format GLM expects.

**Fix:** Set in the model YAML:

```yaml
chat_template_file: config/templates/GLM-4-tool.jinja
```

That template injects tool definitions in GLM’s format and removes the warning. Our GLM configs have this line commented; uncomment if you want native tool description.

