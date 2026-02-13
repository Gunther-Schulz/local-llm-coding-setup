# Model configuration (one YAML per model)

**How it aligns:** Model keys are the YAML filenames (without `.yaml`). Each mode uses its own vars in **config/server.env** (e.g. `PURE_CHAT_MODEL`, `CODING_MODEL`, `NOTEBOOK_CHAT_MODEL`, `CODE_VISION_VISION_MODEL`). Weights live in `models/<model_key>/`. Use the same key in Cursor as the model name. See **MODES.md** for launchers (run_chat.sh, run_coding.sh, run_notebook.sh, run_code_vision.sh).

Each file is `config/models/<model_key>.yaml`. The **model_key** is the filename (without `.yaml`) and must match the directory name under `models/`, i.e. weights live in `models/<model_key>/<gguf>`.

**Source of truth:** only these YAML files. No separate table or `.env` files.

## Schema

**Used by `run_server.sh`** (via `scripts/load_model_config.sh`):  
`gguf`, `mmproj` (optional), `download_url`, `context_size`, `n_gpu_layers`, `jinja`, `temp`, `top_p`, `top_k`, `min_p`, `seed`, `batch_size`, `ubatch_size`.

- **gguf** – GGUF filename or path (e.g. `Qwen3-Coder-Next-MXFP4_MOE.gguf` or `BF16/Qwen3-Coder-Next-BF16-00001-of-00004.gguf`). Full path is `models/<model_key>/<gguf>`.
- **mmproj** – Optional. Vision model projector GGUF filename (e.g. `mmproj-Qwen2.5-VL-7B-Instruct-Q8_0.gguf`). Full path is `models/<model_key>/<mmproj>`. Omit for text-only models.
- **download_url** – Hugging Face repo URL or direct `.../resolve/main/...gguf` URL. Omit or set to `null` for manual-only.
- **context_size** – Runtime context size passed to llama-server as `-c`. Use `0` to use the model’s native max from the GGUF.
- **n_gpu_layers** – `-1` = all on GPU.
- **jinja** – Use Jinja chat template (boolean).
- **temp**, **top_p**, **top_k**, **min_p**, **seed** – Sampling.
- **batch_size**, **ubatch_size** – Batch sizes (optional).
- **chat_template_file** – Optional path to a chat template (e.g. `config/templates/Qwen3-Coder-tool-fix.jinja`). Used by `run_server.sh` only.
- **repeat_penalty** – Optional; e.g. `1.0` to disable (used by GLM-4.7-Flash). Used by `run_server.sh` only.
- **flash_attn**, **cache_type_k**, **cache_type_v** – Optional llama-server flags (e.g. `flash_attn: off`, `cache_type_k: bf16`, `cache_type_v: bf16`). GLM configs have these commented out; uncomment to try reducing CPU load when the model is partly offloaded. See KNOWN_ISSUES.md.

**Used by the chat proxy** (when request `model` matches this file’s key; default: both off):
- **proxy_force_tool_choice_required** – If `true`, set `tool_choice` to `"required"` when the request has tools (avoids grammar trigger bug with GLM).
- **proxy_loop_limits** – If `true`, inject stop after N identical/similar tool calls (uses `PROXY_MAX_IDENTICAL_TOOL_CALLS`, `PROXY_MAX_SIMILAR_TOOL_CALLS` from env).

Set per-mode model keys in `config/server.env` (PURE_CHAT_MODEL, CODING_MODEL, EMBEDDING_MODEL, NOTEBOOK_CHAT_MODEL, CODE_VISION_VISION_MODEL). Mode 4 coding uses CODING_MODEL. Run with `./run_chat.sh`, `./run_coding.sh`, `./run_notebook.sh`, or `./run_code_vision.sh`; or start one server with `./run_server.sh MODEL_KEY [PORT]`.

**Requirements:** `scripts/load_model_config.sh` and `scripts/download-models.sh` need Python 3 and PyYAML (`pip install pyyaml`).
