# Setup build scripts

These scripts are invoked by **setup/install.sh**; they are not meant to be run directly.

- **llamacpp_vision.sh** – Build llama.cpp (CPU) for vision API → `external/llama.cpp/build/`
- **llamacpp_cuda.sh** – Build llama.cpp (CUDA) for LLM engine + benchmarks → `external/llama.cpp/build-cuda/`

To run a full setup including both builds: **./setup/install.sh**

## Native tool-call parsing (llama-server)

Recent llama.cpp (master) includes **native tool-call parsing** for Qwen (Qwen2.5, Qwen3 Coder): when the request has tools, the server parses the model output and returns OpenAI-style `tool_calls` instead of raw text. The proxy then passes them through and does not need to transform.

- The build script **pulls latest** before building so a fresh build gets this.
- If you already have `build-cuda/`, the script skips building. To get native tool support: remove the build dir and rebuild, or run with **FORCE_LLAMACPP_REBUILD=1**:
  - `rm -rf external/llama.cpp/build-cuda && ./setup/build/llamacpp_cuda.sh`
  - or `FORCE_LLAMACPP_REBUILD=1 ./setup/build/llamacpp_cuda.sh`
