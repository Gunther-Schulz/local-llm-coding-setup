# Model configuration (one YAML per model)

**How it aligns:** Model keys are the YAML filenames (without `.yaml`). Each mode uses its own vars in **config/server.env** (e.g. `PURE_CHAT_MODEL`, `CODING_MODEL`, `NOTEBOOK_CHAT_MODEL`). Weights live in `models/<model_key>/`. Use the same key in Cursor as the model name. See **MODES.md** for launchers (run_chat.sh, run_coding.sh, run_notebook.sh).

Each file is `config/models/<model_key>.yaml`. The **model_key** is the filename (without `.yaml`) and must match the directory name under `models/`, i.e. weights live in `models/<model_key>/<gguf>`.

**Source of truth:** only these YAML files. No separate table or `.env` files.

## Schema

**Used by `run_server.sh`** (via `scripts/load_model_config.sh`):  
`gguf`, `download_url`, `context_size`, `n_gpu_layers`, `jinja`, `temp`, `top_p`, `top_k`, `min_p`, `seed`, `batch_size`, `ubatch_size`.

- **gguf** – GGUF filename or path (e.g. `Qwen3-Coder-Next-MXFP4_MOE.gguf` or `BF16/Qwen3-Coder-Next-BF16-00001-of-00004.gguf`). Full path is `models/<model_key>/<gguf>`.
- **download_url** – Hugging Face repo URL or direct `.../resolve/main/...gguf` URL. Omit or set to `null` for manual-only.
- **context_size** – Runtime context size passed to llama-server as `-c`. Use `0` to use the model’s native max from the GGUF.
- **n_gpu_layers** – `-1` = all on GPU.
- **jinja** – Use Jinja chat template (boolean).
- **temp**, **top_p**, **top_k**, **min_p**, **seed** – Sampling.
- **batch_size**, **ubatch_size** – Batch sizes (optional).
- **proxy** – Optional: `compression`, `virtual_tool`, `inject_system`, `inject_capability` (0/1).

Set per-mode model keys in `config/server.env` (PURE_CHAT_MODEL, CODING_MODEL, EMBEDDING_MODEL, NOTEBOOK_CHAT_MODEL). Run with `./run_chat.sh`, `./run_coding.sh`, or `./run_notebook.sh`; or start one server with `./run_server.sh MODEL_KEY [PORT]`.

**Requirements:** `scripts/load_model_config.sh` and `scripts/download-models.sh` need Python 3 and PyYAML (`pip install pyyaml`).
