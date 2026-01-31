# Hardware Corner–style long-context benchmark

Run the Qwen3-Coder 30B Q4_K_XL (GGUF) long-context benchmark locally with llama.cpp + CUDA. Uses the model already in `../models/`.

## Setup

1. **Conda env**

   ```bash
   cd benchmark
   conda env create -f environment.yml
   conda activate benchmark
   ```

2. **Build llama.cpp with CUDA**

   ```bash
   ./build_llamacpp.sh
   ```

   Uses `CUDA_HOME` if set (e.g. `/opt/cuda`); otherwise looks for `nvcc` in common paths.

3. **Run benchmark**

   ```bash
   ./run_benchmark.sh
   ```

   Default model path: `../models/qwen3-coder-30b-a3b-q4_k_xl/qwen3-coder-30b-a3b-instruct-ud-q4_k_xl.gguf`. Override with `BENCHMARK_MODEL=/path/to/model.gguf ./run_benchmark.sh`.

## Model

Same quantization as the Hardware Corner RTX 5090 147K run: Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf (17.7 GB). Must already be present under `../models/`.
