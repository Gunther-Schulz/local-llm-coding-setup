# Chat templates

Templates here override the server’s default (or GGUF-embedded) chat format.

## Qwen3-Coder-tool-fix.jinja

Copy of the Qwen3-Coder chat template with stricter tool-calling instructions:

- **Do NOT omit the initial `<tool_call>` tag** (fixes models that drop it, especially after text).
- **Required parameters MUST be specified and MUST NOT be empty** (reduces empty-args tool calls).

Based on `external/llama.cpp/models/templates/Qwen3-Coder.jinja` plus guidance from [QwenLM/Qwen3-Coder#475](https://github.com/QwenLM/Qwen3-Coder/issues/475).

To revert to the external template, set `chat_template_file: external/llama.cpp/models/templates/Qwen3-Coder.jinja` in the model YAML.
