# Chat templates

Templates here override the server’s default (or GGUF-embedded) chat format.

## GLM-4-tool.jinja

GLM-4.x chat template that **natively describes tools**, so the server injects tool definitions in the format GLM expects instead of using the generic fallback.

- **Tool block**: When `tools` is present, adds a `<|system|>` block with “# Tools” and function signatures inside `<tools></tools>` (each tool as JSON via `tool | tojson(ensure_ascii=False)`).
- **Tool call format**: `<tool_call>{function-name}` with `<arg_key>` / `<arg_value>` pairs; tool responses as `<|observation|>` and `<tool_response>`.
- **Thinking**: Supports `<think></think>` and `enable_thinking` / `clear_thinking`.

Based on `external/llama.cpp/models/templates/GLM-4.6.jinja`. Use for GLM-4.6 / GLM-4.7-Flash (and other GLM-4 GGUF) to remove the warning: “Template supports tool calls but does not natively describe tools.”

Set in model YAML: `chat_template_file: config/templates/GLM-4-tool.jinja`.

### "Grammar still awaiting trigger" (GLM + llama.cpp)

The server uses a **lazy grammar** for tool calls: it only applies the tool-call grammar after the model output matches a trigger (e.g. the start of `<tool_call>`). Until then, each token is checked and the log may show:

```text
Grammar still awaiting trigger after token N (`piece`)
```

- **Benign:** Model is still in preamble, thinking, or text; once it outputs `<tool_call>`, the grammar triggers and tool-call parsing starts. Occasional messages are normal.
- **Bug (issue [#19068](https://github.com/ggml-org/llama.cpp/issues/19068)):** If this repeats for thousands of tokens, output is gibberish, and RAM grows, the server has entered a corrupted state. Seen with GLM-4.7-Flash when using tool definitions (OpenAI format via `--jinja`); trigger is not reliably reproducible.

**Mitigations:** (1) **With proxy:** set `proxy_force_tool_choice_required: true` in the model config so the proxy sends `tool_choice: required` and the grammar is active from the start. (2) **Without proxy:** configure your client (e.g. Cursor) to send `tool_choice: "required"` when using tools with this model, if the client supports it; otherwise the template instructs the model to respond with a tool call first. (3) Avoid malformed prompts (e.g. consecutive user messages without an assistant turn). (4) If corruption occurs: force-kill and restart the server. (5) Run without `--verbose` to reduce log noise from this debug message.

## Qwen3-Coder-tool-fix.jinja

Copy of the Qwen3-Coder chat template with stricter tool-calling instructions:

- **Do NOT omit the initial `<tool_call>` tag** (fixes models that drop it, especially after text).
- **Required parameters MUST be specified and MUST NOT be empty** (reduces empty-args tool calls).

Based on `external/llama.cpp/models/templates/Qwen3-Coder.jinja` plus guidance from [QwenLM/Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475).

To revert to the external template, set `chat_template_file: external/llama.cpp/models/templates/Qwen3-Coder.jinja` in the model YAML.
