#!/usr/bin/env bash
# Thin wrapper for benchmark.py. Passes all arguments through.
# Optional: set CONDA_BENCHMARK_ENV to run under that conda env (e.g. CONDA_BENCHMARK_ENV=llama ./benchmark/next/benchmark.sh --long).
set -e
BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${CONDA_BENCHMARK_ENV:-}" ]]; then
  exec conda run -n "$CONDA_BENCHMARK_ENV" python3 "$BENCH_DIR/benchmark.py" "$@"
fi
exec python3 "$BENCH_DIR/benchmark.py" "$@"
