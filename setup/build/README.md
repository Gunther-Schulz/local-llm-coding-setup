# Setup build scripts

These scripts are invoked by **setup/install.sh**; they are not meant to be run directly.

- **llamacpp_vision.sh** – Build llama.cpp (CPU) for vision API → `external/llama.cpp/build/`
- **llamacpp_cuda.sh** – Build llama.cpp (CUDA) for LLM engine + benchmarks → `external/llama.cpp/build-cuda/`

To run a full setup including both builds: **./setup/install.sh**
