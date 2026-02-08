# Setup build scripts

These scripts are invoked by **setup/install.sh**; they can also be run directly when needed.

- **llamacpp_vision.sh** – Build llama.cpp (CPU) for vision API → `external/llama.cpp/build/`
- **llamacpp_cuda.sh** – Build llama.cpp (CUDA) for LLM engine + benchmarks → `external/llama.cpp/build-cuda/`
- **update_llamacpp.sh** – Pull latest llama.cpp from master and rebuild both vision and CUDA (one-command update)

To run a full setup including both builds: **./setup/install.sh**

## Updating llama.cpp to the newest version

To update to the latest llama.cpp and rebuild:

```bash
./setup/build/update_llamacpp.sh
```

This pulls from `origin master` and force-rebuilds both vision and CUDA. Optional env vars:

- `LLAMACPP_UPDATE_VISION=0` – skip vision rebuild
- `LLAMACPP_UPDATE_CUDA=0` – skip CUDA rebuild

Manual options (if you only need one build):

- **CUDA only:** `FORCE_LLAMACPP_REBUILD=1 ./setup/build/llamacpp_cuda.sh` (script already pulls when run)
- **Vision only:** `FORCE_LLAMACPP_REBUILD=1 ./setup/build/llamacpp_vision.sh` (script pulls when run)

## Native tool-call parsing (llama-server)

Recent llama.cpp (master) includes **native tool-call parsing** for Qwen (Qwen2.5, Qwen3 Coder): when the request has tools, the server parses the model output and returns OpenAI-style `tool_calls` instead of raw text. The proxy then passes them through and does not need to transform.

- Both build scripts **pull latest** before building when the repo already exists.
- If binaries already exist, the scripts skip building unless you use **FORCE_LLAMACPP_REBUILD=1** or run **update_llamacpp.sh**.
