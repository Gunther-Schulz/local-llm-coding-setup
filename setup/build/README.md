# Setup build scripts

Scripts to build and update **llama.cpp**. Used by `./setup/install.sh`; can be run directly when needed.

- **llamacpp_cuda.sh** – Build llama.cpp (CUDA) for LLM server → `external/llama.cpp/build-cuda/`. If `LLAMACPP_BRANCH` is set, checks out that branch instead of master.
- **llamacpp_vision.sh** – (Optional) Build llama.cpp (CPU) for vision → `external/llama.cpp/build/`
- **update_llamacpp.sh** – Pull latest master and rebuild CUDA only
- **rebuild_llamacpp_glm_fix.sh** – Check out `fix/glm45-tool-parse-only-auto` and rebuild CUDA (GLM 4.5 tool-call fix). Override branch: `LLAMACPP_GLM_FIX_BRANCH=my-branch ./setup/build/rebuild_llamacpp_glm_fix.sh`

## First-time setup

```bash
./setup/install.sh
```

Creates conda env `vLLM`, uses `.wheels/` as pip cache, and builds llama.cpp CUDA.

## Update llama.cpp

To pull latest and rebuild (e.g. for grammar/tool-call fixes):

```bash
./setup/build/update_llamacpp.sh
```

- Rebuilds CUDA only (llama-server).

Manual rebuild (no git pull): `FORCE_LLAMACPP_REBUILD=1 ./setup/build/llamacpp_cuda.sh`

Build from GLM 4.5 fix branch (parse-only for tool_choice=AUTO): `./setup/build/rebuild_llamacpp_glm_fix.sh`

## Wheel cache

`./setup/install.sh` sets `PIP_CACHE_DIR` to `.wheels/` (or `WHEEL_CACHE` if set). Reinstalls reuse cached wheels. Override: `WHEEL_CACHE=/path ./setup/install.sh` or `PIP_CACHE_DIR=/path pip install ...`.
