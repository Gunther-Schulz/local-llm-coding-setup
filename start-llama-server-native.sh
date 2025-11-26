#!/bin/bash
cd /workspace/llama.cpp-native/build/bin

./llama-server \
  --model /workspace/models/qwen2.5-coder-14b-q4_k_m/qwen2.5-coder-14b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --ctx-size 81920 \
  --rope-scale 2.5 \
  --no-context-shift \
  --threads 16 \
  --n-gpu-layers -1 \
  --batch-size 256 \
  --ubatch-size 256 \
  --parallel 1 \
  --flash-attn on \
  --cont-batching \
  --jinja \
  --cache-ram 0

