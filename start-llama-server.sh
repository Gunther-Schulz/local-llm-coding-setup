#!/bin/bash
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate glm4
cd /workspace
python3 -m llama_cpp.server \
  --model /workspace/models/qwen2.5-coder-14b-q4_k_m/qwen2.5-coder-14b-instruct-q4_k_m.gguf \
  --host 0.0.0.0 \
  --port 8000 \
  --n_ctx 81920 \
  --n_threads 8 \
  --n_gpu_layers -1 \
  --n_batch 128 \
  --n_ubatch 128
