# Hardware Corner–style long-context benchmark

Run the Qwen3-Coder 30B Q4_K_XL (GGUF) long-context benchmark locally with llama.cpp + CUDA. Uses the model in `../models/`.

## Setup

1. **Build llama.cpp (CUDA)**  
   From project root:
   ```bash
   ./setup/install.sh
   ```
   (Builds vLLM + llama.cpp vision + CUDA; binaries in `external/llama.cpp/build-cuda/`.)  
   To build only CUDA: `./setup/build/llamacpp_cuda.sh`.

2. **Run benchmark**  
   From project root or from `benchmark/`:
   ```bash
   ./benchmark/run_benchmark.sh
   ```
   Default model: `../models/qwen3-coder-30b-a3b-q4_k_xl/qwen3-coder-30b-a3b-instruct-ud-q4_k_xl.gguf`. Override: `BENCHMARK_MODEL=/path/to/model.gguf ./benchmark/run_benchmark.sh`.

Use the project’s main conda env (from `./setup/install.sh`). No separate env for the benchmark.

## Model

Same quantization as the Hardware Corner RTX 5090 147K run: Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf (17.7 GB). Must already be present under `../models/`.

## Layout

- **external/llama.cpp/** – Single llama.cpp clone (shared by vision and CUDA). Built by `./setup/install.sh`.
- **external/llama.cpp/build-cuda/** – CUDA build used by this benchmark and by `./run/run llm` when engine=llamacpp.
- **benchmark/** – Scripts and config only (no llama.cpp clone here).
