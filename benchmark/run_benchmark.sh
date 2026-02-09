#!/usr/bin/env bash
# Run Hardware Corner–style benchmark: llama-bench with their exact methodology
# (Flash Attention, context depths, PP 1024 / TG 128, all layers on GPU)
# Optional: BENCHMARK_MODEL=... BENCHMARK_DEPTHS=... (Fish: use env BENCHMARK_MODEL=... ./run_benchmark.sh)
set -e

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$BENCH_DIR/.." && pwd)"

BENCHMARK_MODEL="${BENCHMARK_MODEL:-$ROOT/models/qwen3-coder-30b-a3b-q4_k_xl/qwen3-coder-30b-a3b-instruct-ud-q4_k_xl.gguf}"
LLAMA_BENCH="${LLAMA_BENCH:-$ROOT/external/llama.cpp/build-cuda/bin/llama-bench}"

if [ ! -f "$BENCHMARK_MODEL" ]; then
  echo "Model not found: $BENCHMARK_MODEL"
  echo "Set BENCHMARK_MODEL=/path/to/model.gguf or place model in $ROOT/models/..."
  exit 1
fi

if [ ! -x "$LLAMA_BENCH" ]; then
  echo "llama-bench not found: $LLAMA_BENCH"
  echo "Run ./setup/install.sh (includes llama.cpp CUDA) or ./setup/build/llamacpp_cuda.sh"
  exit 1
fi

# Hardware Corner methodology: -fa 1, -d context depths, -p 1024, -n 128, -ngl 99
# Context depths: standard sweep + 147K for 30B MoE extreme
DEPTHS="${BENCHMARK_DEPTHS:-4098,8196,16384,32768,45062,57356,65536,86026,131072,147000}"

echo "Model: $BENCHMARK_MODEL"
echo "Depths: $DEPTHS"
echo ""

"$LLAMA_BENCH" \
  -m "$BENCHMARK_MODEL" \
  -fa 1 \
  -d "$DEPTHS" \
  -p 1024 \
  -n 128 \
  -ngl 99
