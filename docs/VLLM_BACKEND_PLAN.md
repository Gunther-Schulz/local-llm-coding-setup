# Plan: vLLM as alternative backend

**Goal:** Support vLLM as an alternative to llama-server so that a model can be served by either backend. Same UX: set `CODING_MODEL=glm-4.7-flash-ud-q6-k-xl` in `config/server.env`, run `./run_coding.sh` — backend is chosen from the model’s YAML (llama vs vLLM). No change to launchers or proxy from the user’s perspective.

**Principles:** Single entry point (`run_server.sh`), backend per model (in YAML), minimal changes to `config/server.env`, shared config where both backends support the same concept (e.g. temp, top_p). **Use the existing project setup:** conda env **vLLM**, no new venv.

---

## 0. Current setup (verified)

- **Conda env:** The project uses a single conda env named **vLLM** (Python 3.10), created and used by `./setup/install.sh`. There is no venv; scripts and docs should refer to this conda env, not “create a venv” or “VLLM_PYTHON to a venv”.
- **setup/install.sh:** Sources conda, creates/verifies env `vLLM`, activates it, sets `PIP_CACHE_DIR` to `.wheels/` (or `WHEEL_CACHE`), runs `pip install -U pip`, installs from `requirements.txt` if present (there is no root `requirements.txt` today), then builds llama.cpp CUDA. Post-install message: “Activate: conda activate vLLM”, “Next: ./run_server.sh [PORT]”.
- **run_server.sh:** Does **not** activate conda; it runs the llama-server binary directly. Launchers (run_chat.sh, run_coding.sh, etc.) call `run_server.sh`; the user may or may not have run `conda activate vLLM` in the same shell.
- **start-proxy.sh:** Runs `python3 scripts/chat_proxy.py` — uses whatever `python3` is on PATH (if user activated vLLM, that’s the one).
- **benchmark/next/benchmark.sh:** Optional `CONDA_BENCHMARK_ENV` to run benchmark.py under a conda env (e.g. `CONDA_BENCHMARK_ENV=vLLM`).
- **config:** No conda or vLLM env name in `config/server.env` today. All mode/model config is in server.env + config/models/*.yaml.
- **Implication for vLLM backend:** When we start the vLLM server, we must run it **inside the vLLM conda env**. The robust way is to invoke it via `conda run -n vLLM vllm serve ...` (or a configurable env name from server.env) so it works even when the user has not activated vLLM before running `./run_coding.sh`. Relying on “user must activate vLLM first” is possible but fragile.

---

## 1. Architecture overview

- **Entry point:** `./run_server.sh [--verbose] MODEL_KEY [PORT]` (unchanged). Used by `run_chat.sh`, `run_coding.sh`, and the two servers in `run_code_vision.sh`.
- **Backend selection:** After loading `config/server.env` and `config/models/<MODEL_KEY>.yaml`, backend is read from the model YAML (`backend: llama | vllm`). Default is `llama` if omitted.
- **Execution:**
  - If `backend == llama`: current behavior (llama-server, existing argv from YAML + server.env).
  - If `backend == vllm`: run a vLLM runner that builds `vllm serve ...` from the same YAML (vLLM-specific keys) and shared keys (port, host, temp, top_p, etc.).
- **Proxy and Cursor:** No change. Proxy still uses `BACKEND_URL` (default `http://127.0.0.1:8001`). Whatever server is on that port (llama or vLLM) is what the proxy forwards to.
- **run_notebook.sh:** Stays llama-server router only (multi-model preset). vLLM support in this plan is for single-model modes (chat, coding, code_vision). Notebook + vLLM can be a later extension.

---

## 2. config/server.env

- **No structural change.** Keep mode vars: `PURE_CHAT_MODEL`, `PURE_CHAT_PORT`, `CODING_MODEL`, `CODING_PORT`, `HOST`, `LLAMA_THREADS`, `CURSOR_MODEL_ALIAS`, etc.
- **vLLM backend:** Use the existing **conda env vLLM**. Add an optional var:
  - **`VLLM_CONDA_ENV`** (optional): conda env name to run `vllm serve` in. Default: `vLLM`. The vLLM runner will use `conda run -n "$VLLM_CONDA_ENV" vllm serve ...` so the server runs in that env even if the user did not activate it. No venv, no `VLLM_PYTHON` — we use conda.
- **Comment:** Add one line that backend is per-model and defined in `config/models/<key>.yaml` (e.g. `backend: vllm`).

---

## 3. config/models/*.yaml — schema

**Backend-agnostic (used by proxy / both backends):**

- `download_url` — keep for docs/download scripts; vLLM may load from HF by model id instead.
- `proxy_force_tool_choice_required`, `proxy_loop_limits` — unchanged; proxy reads these when request model matches.

**Backend selector (single YAML, one active backend):**

- **`backend`** (optional): `llama` | `vllm`. Default: `llama`. Only one backend is active per model; the runner uses the block for that backend and ignores the other.
- **Single file, one backend:** One YAML can have `backend: vllm` and vLLM-only keys, or `backend: llama` (default) and llama-only keys. User switches backend by editing this one key (or by overriding via env if we add that). No need for two separate files.
- **Optional: nested blocks** (future): We could support `llamacpp: { gguf: ..., ... }` and `vllm: { model: ..., tool_call_parser: glm47 }` in the same file, with top-level `backend: llama | vllm` choosing which block to use. That keeps both backends’ config in one place. For the first version, flat keys (`backend` + `vllm_*` vs `gguf`, etc.) are enough; nested blocks can be added later if desired.
- **No problem** with one YAML and a single `backend` key: the loader exports only the vars for the selected backend; the runner uses them. No technical downside.

**Llama-only (existing; ignored when backend=vllm):**

- `gguf`, `mmproj`, `context_size`, `n_gpu_layers`, `threads`, `jinja`, `temp`, `top_p`, `top_k`, `min_p`, `repeat_penalty`, `seed`, `batch_size`, `ubatch_size`, `chat_template_file`, `flash_attn`, `cache_type_k`, `cache_type_v`.

**vLLM-only (ignored when backend=llama):**

- **`vllm_model`** (required when backend=vllm): Model to load. Can be: (1) HuggingFace model id (e.g. `zai-org/GLM-4.7-Flash`), (2) local path to a GGUF file (e.g. `./models/glm-4.7-flash-ud-q6-k-xl/GLM-4.7-Flash-UD-Q6_K_XL.gguf`), or (3) HF GGUF repo with quant type: `repo_id:quant_type` (e.g. `unsloth/GLM-4.7-Flash-GGUF:Q4_K_M`). Passed as first positional to `vllm serve <vllm_model>`. For local GGUF, vLLM recommends `--tokenizer <base-model-name>` (see “vLLM and GGUF” below).
- **`vllm_tool_call_parser`** (optional): e.g. `glm47` for GLM-4.7 tool calling. Maps to `--tool-call-parser`.
- **`vllm_max_model_len`** (optional): e.g. `202752`. Maps to `--max-model-len`.
- **`vllm_tensor_parallel`** (optional): e.g. `1`. Maps to `--tensor-parallel-size`.
- **`vllm_gpu_memory_utilization`** (optional): e.g. `0.9`. Maps to `--gpu-memory-utilization`.
- **`vllm_serve_extra`** (optional): string or list of extra args for `vllm serve` (e.g. `--enable-auto-tool-choice`). If list, join with space.
- **`vllm_tokenizer`** (optional): tokenizer model name or path (e.g. `zai-org/GLM-4.7-Flash`). Maps to `--tokenizer`. Recommended when using a local GGUF so vLLM uses the base model’s tokenizer instead of extracting from GGUF (faster and more stable).

**Shared where possible:**

- For vLLM runner, map **`temp`**, **`top_p`** from YAML to vLLM’s sampling args so the same YAML can drive both backends for sampling. vLLM uses similar flags; document the mapping in README.

**Example vLLM model YAML (same logical model, switch backend in one file):**

```yaml
# Same key as llama version; switch backend here to use vLLM instead.
backend: vllm
vllm_model: zai-org/GLM-4.7-Flash   # HF safetensors; or local GGUF path; or unsloth/GLM-4.7-Flash-GGUF:Q6_K
vllm_tool_call_parser: glm47
vllm_max_model_len: 202752
vllm_tokenizer: zai-org/GLM-4.7-Flash   # optional; recommended when using GGUF (see below)
temp: 0.7
top_p: 1.0
proxy_force_tool_choice_required: true
proxy_loop_limits: false
```

User switches backend by editing this file (`backend: llama` vs `backend: vllm`) or by having two YAMLs (e.g. `glm-4.7-flash-ud-q6-k-xl.yaml` for llama, `glm-4.7-flash-vllm.yaml` for vLLM) and setting `CODING_MODEL` in server.env. Single YAML with one `backend` key is sufficient and avoids duplicate files.

---

## 3a. vLLM and GGUF (research summary)

**vLLM does support GGUF**, but with caveats:

- **Status:** Documented as “highly experimental and under-optimized”; may be incompatible with some features. Primary use case is reducing memory via quantization.
- **Single-file only:** Only single-file GGUF is supported. Multi-file GGUF must be merged (e.g. with gguf-split) first.
- **Loading:**  
  - From HuggingFace: `repo_id:quant_type` (e.g. `unsloth/Qwen3-0.6B-GGUF:Q4_K_M`).  
  - Local file: `vllm serve ./path/to/model.gguf --tokenizer <base-model>`.
- **Tokenizer:** vLLM recommends using the **tokenizer from the base model** (e.g. `--tokenizer Qwen/Qwen3-0.6B` or `zai-org/GLM-4.7-Flash`) rather than extracting from the GGUF, because conversion from GGUF is slow and unstable for large vocabs. So when using a local GGUF in our setup, we should support `vllm_tokenizer` in the YAML and pass it as `--tokenizer` to `vllm serve`.
- **Supported quants:** Standard (Q4_0, Q5_0, Q8_0, etc.), K-quants (Q2_K–Q6_K), and some imatrix types.

**Conclusion for our plan:** We can use GGUF with vLLM (local path or `repo_id:quant_type`). Expose `vllm_tokenizer` in the model YAML and pass it to vLLM when set. Document that GGUF on vLLM is experimental; for production GLM tool calling, HF safetensors + vLLM may be more stable than GGUF + vLLM.

---

## 3b. vLLM install (no compile) — and how it fits our setup

**You do not need to compile vLLM.** Use pre-built wheels. **Our project already has the conda env:** `vLLM` (Python 3.10), created by `./setup/install.sh`. We use that env for the vLLM backend; no separate venv.

- **Where to install vllm:** In the existing **vLLM** conda env. After `./setup/install.sh`, user runs:
  - `conda activate vLLM`
  - `pip install vllm`  
  or with a specific CUDA index if needed:  
  `pip install vllm --extra-index-url https://download.pytorch.org/whl/cu129`
- **Optional:** Add a root `requirements.txt` that includes `vllm` (or a `requirements-vllm.txt` and a step in `setup/install.sh` that installs it when present), so `./setup/install.sh` installs vllm into the vLLM env in one go. Otherwise document: “For vLLM backend: conda activate vLLM && pip install vllm”.
- **Runner:** The vLLM runner script does **not** assume the user has activated conda. It runs vllm via **`conda run -n vLLM vllm serve ...`** (or `conda run -n "$VLLM_CONDA_ENV" ...` if set in server.env), so the correct env is used regardless of the current shell. No `VLLM_PYTHON` or venv path — we rely on the vLLM conda env.
- **Wheel cache:** `./setup/install.sh` already sets `PIP_CACHE_DIR` to `.wheels/`; pip install vllm in the vLLM env will use that cache.
- **No compilation:** Pre-built wheels; no build-from-source step.

---

## 4. scripts/load_model_config.sh

- **New mappings** (YAML key → env var):
  - `backend` → `BACKEND` (default `llama` if omitted).
  - `vllm_model` → `VLLM_MODEL`
  - `vllm_tool_call_parser` → `VLLM_TOOL_CALL_PARSER`
  - `vllm_max_model_len` → `VLLM_MAX_MODEL_LEN`
  - `vllm_tensor_parallel` → `VLLM_TENSOR_PARALLEL`
  - `vllm_gpu_memory_utilization` → `VLLM_GPU_MEMORY_UTILIZATION`
  - `vllm_serve_extra` → `VLLM_SERVE_EXTRA` (string; if list in YAML, join with space in Python before export)
  - `vllm_tokenizer` → `VLLM_TOKENIZER` (optional; for GGUF, pass as `--tokenizer` to vllm serve)
- **Existing mappings:** Unchanged. Llama path in `run_server.sh` keeps using `GGUF`, `CONTEXT_SIZE`, etc. vLLM path will use `VLLM_*` and optionally `TEMP`, `TOP_P` from existing exports.

---

## 5. run_server.sh — dispatcher

- After loading `server.env` and model config (eval of `load_model_config.sh`):
  1. **If `BACKEND == vllm`** (or `VLLM_MODEL` is set and non-empty):  
     - Call a **vLLM runner** (see below) with: `ACTIVE_MODEL`, `PORT`, `HOST`, `CURSOR_MODEL_ALIAS`, and all `VLLM_*` and shared (TEMP, TOP_P) env vars.  
     - Runner is responsible for building and `exec`-ing `vllm serve ...`.  
     - Do not run llama-server or build llama argv in this branch.
  2. **Else:**  
     - Keep current logic: build llama-server argv from existing env, `exec "$LLAMA_SERVER" "${argv[@]}"`.
- **Log file:** Today we have `SERVER_LOG`, `rm -f "$SERVER_LOG"`, and `--log-file "$SERVER_LOG"` for llama. For vLLM, either:
  - Redirect vLLM process stdout/stderr to `SERVER_LOG` (e.g. `exec ... >> "$SERVER_LOG" 2>&1`), or
  - Use vLLM’s own logging if it has a `--log-file`-style option; then set a similar path (e.g. `logs/server.log` or `logs/vllm-server.log`) so “one log per run, overwritten” behavior is consistent.
- **Startup message:** Print which backend is used (e.g. “backend=llama” or “backend=vllm”) and port/model so the user sees it in the terminal.

---

## 6. vLLM runner (new script or inline)

**Recommendation:** New script **`scripts/run_vllm.sh`** (or `scripts/start_vllm.sh`) that:

- **Inputs:** All from environment (set by `run_server.sh` after loading server.env and model config): `VLLM_MODEL`, `PORT`, `HOST`, `CURSOR_MODEL_ALIAS`, `VLLM_CONDA_ENV` (default `vLLM` from server.env or literal `vLLM`), `VLLM_TOOL_CALL_PARSER`, `VLLM_MAX_MODEL_LEN`, `VLLM_TENSOR_PARALLEL`, `VLLM_GPU_MEMORY_UTILIZATION`, `VLLM_SERVE_EXTRA`, `VLLM_TOKENIZER`, `TEMP`, `TOP_P`, `ACTIVE_MODEL`, `SERVER_LOG`.
- **Behavior:**
  1. **Resolve vllm command:** Run vllm **inside the vLLM conda env** via `conda run -n "$VLLM_CONDA_ENV" vllm serve ...` (so we do not depend on the user having run `conda activate vLLM` in the same shell). If conda is not available or the env does not exist, exit with a clear message (“Activate conda and run ./setup/install.sh” or “Install vllm in the vLLM env: conda activate vLLM && pip install vllm”).
  2. Build argv: `--host`, `--port`, model as first positional (`$VLLM_MODEL`). Add `--tool-call-parser "$VLLM_TOOL_CALL_PARSER"` if set; `--max-model-len` if set; `--tensor-parallel-size`, `--gpu-memory-utilization` if set. Map `TEMP`/`TOP_P` to vLLM flags if vLLM supports them (e.g. `--temperature`, `--top-p`). Append `$VLLM_SERVE_EXTRA` split by space.
  3. **Model name in API:** So Cursor sees a stable name, use vLLM’s way to set the served model name (e.g. `--served-model-name` or whatever vLLM has); set to `CURSOR_MODEL_ALIAS` or `ACTIVE_MODEL` so it matches what users put in Cursor.
  4. **Logging:** Either `exec` with stdout/stderr to `SERVER_LOG`, or vLLM’s own log file. Prefer one log file per run, overwritten (same as llama).
  5. **Exec:** `exec` the vLLM process so it replaces the shell (same as llama-server).
- **Error handling:** If `VLLM_MODEL` is empty, print usage and exit 1. If conda or the vLLM env is missing, or `vllm` is not installed in that env, exit with a clear message: e.g. “Run ./setup/install.sh and install vllm in the vLLM env: conda activate vLLM && pip install vllm”.

**Alternative:** Inline the vLLM argv building inside `run_server.sh` in a block “if BACKEND=vllm; then ... exec vllm serve ...; fi”. A separate script keeps `run_server.sh` shorter and makes it easy to test vLLM args independently.

---

## 7. Launchers (run_chat.sh, run_coding.sh, run_code_vision.sh)

- **No code change.** They already do `exec "$ROOT/run_server.sh" "$CODING_MODEL" "$CODING_PORT"` (or equivalent). Backend is determined by the model key’s YAML.
- **Usability:** User switches backend by changing the model key in `config/server.env` (e.g. `CODING_MODEL=glm-4.7-flash-vllm`) or by editing the same key’s YAML to `backend: vllm` (Option A). Option B is clearer: two keys, two YAMLs.

---

## 8. Proxy and start-proxy.sh

- **No change.** Proxy continues to use `BACKEND_URL` (default `http://127.0.0.1:8001`). Whichever server is started on that port (llama or vLLM) is what the proxy forwards to. OpenAI-compatible API is assumed for both backends.

---

## 9. run_code_vision.sh and run_notebook.sh

- **run_code_vision.sh:** Already runs two `run_server.sh` invocations (vision model + coding model). Both will automatically use vLLM if the corresponding model YAML has `backend: vllm`. No change to the script.
- **run_notebook.sh:** Currently uses llama-server with a router preset (multiple models, INI). Stays llama-only in this plan. Adding vLLM to notebook mode (e.g. chat model on vLLM) would be a separate, later step.

---

## 10. Setup / install

- **Existing:** The project already has the **vLLM** conda env (Python 3.10) from `./setup/install.sh`. No new venv; we use this env for the vLLM backend.
- **vLLM package:** Ensure the vLLM env has the `vllm` package. Either:
  - Add `vllm` to a root `requirements.txt` (or a dedicated `requirements-vllm.txt`) and have `setup/install.sh` install it when that file exists, or
  - Document: “For vLLM backend: after ./setup/install.sh, run `conda activate vLLM && pip install vllm`.”
- **Optional:** A small `setup/install_vllm.sh` that activates the vLLM env and runs `pip install vllm` (and optionally a specific CUDA index) so users have a one-command way to add vllm to the existing env. Not required for the plan; convenience only.
- **Config:** Optional `VLLM_CONDA_ENV=vLLM` in `config/server.env` (default vLLM) so the runner knows which conda env to use. No `VLLM_PYTHON` or venv paths.

---

## 11. Documentation and usability

- **config/models/README.md:** Extend schema section:
  - Document `backend: llama | vllm`.
  - Document all `vllm_*` keys and that they apply only when `backend: vllm`.
  - Note that for vLLM, `vllm_model` is the HuggingFace model id or path (not GGUF filename).
  - Give one example YAML for a vLLM model (e.g. GLM-4.7-Flash with `vllm_tool_call_parser: glm47`).
- **config/server.env:** Add a short comment that backend is per-model (see `config/models/<key>.yaml`).
- **Main README or MODES.md:** One sentence: “You can serve a model with llama-server or vLLM; set `backend: vllm` and `vllm_model` in the model’s YAML and use the same launchers.”

---

## 12. Edge cases and details

- **Port and host:** Same as today: from `server.env` (e.g. `CODING_PORT=8001`, `HOST=127.0.0.1`). vLLM runner receives `PORT` and `HOST` and passes them to `vllm serve`.
- **Model name in responses:** vLLM should expose the same “model” name as Cursor expects (e.g. `CURSOR_MODEL_ALIAS` or the model key). Use vLLM’s option (e.g. `--served-model-name`) so /v1/models and chat responses show that name.
- **Log file:** Reuse `logs/server.log` and overwrite on each start for vLLM too (redirect or vLLM’s flag), so behavior is consistent with llama.
- **Failure if vLLM not installed:** Runner checks that conda and the vLLM env exist and that `conda run -n vLLM vllm --help` (or similar) succeeds; if not, exit with a clear message: e.g. “Install vllm in the vLLM conda env: conda activate vLLM && pip install vllm”.

---

## 13. Implementation order (when you code)

1. **Schema and config**
   - Extend `scripts/load_model_config.sh` with `backend` and `vllm_*` → `BACKEND`, `VLLM_*` exports.
   - Add one example vLLM model YAML (e.g. `config/models/glm-4.7-flash-vllm.yaml`) and document in `config/models/README.md`.
2. **vLLM runner**
   - Implement `scripts/run_vllm.sh` (or chosen name) with argv building and `exec`, log file handling, and model name.
3. **run_server.sh**
   - After loading model config, branch on `BACKEND`: if vllm, call the vLLM runner (source it or exec it with env passed); else keep current llama-server block. Add “backend=…” to startup echo.
4. **server.env and docs**
   - Optional `VLLM_CONDA_ENV` in server.env (default vLLM) and comment about backend in model YAML. Update README/MODES.md and config/models/README.md as above.
5. **Manual test**
   - Set `CODING_MODEL=glm-4.7-flash-vllm` (or the key you added), run `./run_coding.sh`, confirm vLLM starts and Cursor/proxy can talk to it. Then switch back to a llama model and confirm llama still works.

This gives you a single entry point, backend chosen per model in YAML, minimal server.env change, and the same launcher/proxy UX for both backends.
