# Model configuration (one YAML per model)

**How it aligns:** Model keys are the YAML filenames (without `.yaml`). Each mode uses its own vars in **config/server.env** (e.g. `PURE_CHAT_MODEL`, `CODING_MODEL`, `NOTEBOOK_CHAT_MODEL`, `CODE_VISION_VISION_MODEL`). Weights live in `models/<model_key>/`. Use the same key in Cursor as the model name. See **MODES.md** for launchers (run_chat.sh, run_coding.sh, run_notebook.sh, run_code_vision.sh).

Each file is `config/models/<model_key>.yaml`. The **model_key** is the filename (without `.yaml`) and must match the directory name under `models/`, i.e. weights live in `models/<model_key>/<gguf>`.

**Source of truth:** only these YAML files. No separate table or `.env` files.

## Schema

Config can use **nested YAML** (recommended) or **flat** keys. The loader supports both.

**Nested (recommended):** Use `backend` plus two dicts: `llama: { ... }` and `vllm: { ... }`. Shared keys (e.g. `temp`, `top_p`, `proxy_*`) stay at top level. Example:

```yaml
backend: llama   # or vllm
temp: 0.7
top_p: 1.0
proxy_force_tool_choice_required: true

llama:
  gguf: My-Model.gguf
  context_size: 0
  n_gpu_layers: -1
  jinja: true
  repeat_penalty: 1.0
  # ...

vllm:
  model: models/my-model/My-Model.gguf   # or HuggingFace id
  tool_call_parser: glm47
  max_model_len: 202752
  tokenizer: zai-org/GLM-4.7-Flash
```

**Flat (legacy):** All keys at top level: `backend`, `gguf`, `context_size`, … and when vLLM: `vllm_model`, `vllm_tool_call_parser`, etc. Still supported.

**Used by `run_server.sh`** (via `scripts/load_model_config.sh`):  
`backend` (optional), top-level shared: `temp`, `top_p`, `top_k`, `min_p`, `proxy_*`. Under `llama:` (or flat): `gguf`, `mmproj`, `context_size`, `n_gpu_layers`, `jinja`, `repeat_penalty`, `seed`, `batch_size`, `ubatch_size`, `chat_template_file`, `flash_attn`, `cache_type_k`, `cache_type_v`. Under `vllm:` (or flat `vllm_*`): `model`, `tool_call_parser`, `max_model_len`, `tensor_parallel`, `gpu_memory_utilization`, `serve_extra`, `tokenizer`.

- **backend** – `llama` (default) | `vllm`. Chooses which server runs. When `vllm`, the model YAML must provide vLLM config (under `vllm:` or flat `vllm_model`, etc.).
- **llama** (dict) – Llama-server–only options: `gguf`, `mmproj`, `context_size`, `n_gpu_layers`, `jinja`, `repeat_penalty`, `seed`, `batch_size`, `ubatch_size`, `chat_template_file`, `flash_attn`, `cache_type_k`, `cache_type_v`.
- **vllm** (dict) – vLLM-only options: `model`, `tool_call_parser`, `max_model_len`, `tensor_parallel`, `gpu_memory_utilization`, `serve_extra`, `tokenizer`.
- **gguf** (under `llama` or flat) – GGUF filename or path. Full path is `models/<model_key>/<gguf>`. Ignored when `backend: vllm`.
- **mmproj** (under `llama` or flat) – Optional. Vision model projector GGUF filename. Full path is `models/<model_key>/<mmproj>`.
- **download_url** – Hugging Face repo URL (or direct `.../resolve/main/...gguf` link). Top-level. Used by `scripts/download-models.sh` to fetch the main GGUF and any extra files.
- **download_extra** – Optional. List of additional filenames to download from the same repo, e.g. `[ "mmproj-model-Q8_0.gguf" ]` or `[ "subdir/tokenizer.json" ]`. The downloader also auto-includes `llama.mmproj` when set (vision models). Same repo as `download_url`; paths are relative to repo root.
- **context_size**, **n_gpu_layers**, **jinja**, **repeat_penalty**, **seed**, **batch_size**, **ubatch_size**, **chat_template_file**, **flash_attn**, **cache_type_k**, **cache_type_v** – Under `llama` (or flat). See KNOWN_ISSUES.md for GLM + llama.cpp.
- **temp**, **top_p**, **top_k**, **min_p** – Shared; top-level. Sampling.
- **model** (under `vllm` or flat `vllm_model`) – Model to load: HuggingFace id, local GGUF path, or `repo_id:quant_type`. Passed to `vllm serve`.
- **tool_call_parser** (under `vllm` or flat `vllm_tool_call_parser`) – e.g. `glm47` for GLM-4.7. Maps to `--tool-call-parser`.
- **max_model_len**, **tensor_parallel**, **gpu_memory_utilization**, **serve_extra**, **tokenizer** – Under `vllm` (or flat `vllm_*`). Optional.

**Used by the chat proxy** (when request `model` matches this file’s key; default: both off):
- **proxy_force_tool_choice_required** – If `true`, set `tool_choice` to `"required"` when the request has tools (avoids grammar trigger bug with GLM).
- **proxy_loop_limits** – If `true`, inject stop after N identical/similar tool calls (uses `PROXY_MAX_IDENTICAL_TOOL_CALLS`, `PROXY_MAX_SIMILAR_TOOL_CALLS` from env).

Set per-mode model keys in `config/server.env` (PURE_CHAT_MODEL, CODING_MODEL, EMBEDDING_MODEL, NOTEBOOK_CHAT_MODEL, CODE_VISION_VISION_MODEL). Mode 4 coding uses CODING_MODEL. Run with `./run_chat.sh`, `./run_coding.sh`, `./run_notebook.sh`, or `./run_code_vision.sh`; or start one server with `./run_server.sh MODEL_KEY [PORT]`.

**Requirements:** `scripts/load_model_config.sh` and `scripts/download-models.sh` need Python 3 and PyYAML (`pip install pyyaml`).
