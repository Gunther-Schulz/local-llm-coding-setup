# Qwen3-Coder-Next MoE / KV-cache benchmark

Standalone llama.cpp testing (no proxy): compare Q2/Q3/Q4 with MoE offload and KV cache quantization to measure performance penalty on your 5090 + 64GB RAM.

## Layout

- **download.sh** – Download Q2, Q3, Q4 GGUF from Unsloth (HuggingFace).
- **scenarios.cfg** – Scenario definitions: `name|model_path|moe_ot|cache_k|n_gpu_layers`. Optional **n_gpu_layers**: empty = all GPU; `0` = all CPU; integer = that many layers on GPU; `25%` / `50%` / `75%` = share on GPU (uses `BENCHMARK_N_LAYERS=80` for this model; override with env if needed).
- **run_server.sh** – Start llama-server for one scenario (port 18999).
- **fill_context.sh** – Build a long prompt from project files (proxy/ + stack/) for long-context tests.
- **measure.py** – Call `/v1/chat/completions`, report tokens and tok/s.
- **benchmark.sh** – Run each scenario: start server → short prompt (default) or short + long with `--long`; print table. Writes **results.txt** with tok/s and **context used** (prompt tokens). By default runs GPU then CPU (system-only) pass; set `RUN_CPU=0` to skip CPU pass.

## Prereqs

- llama-server (CUDA): `./external/llama.cpp/build-cuda/bin/llama-server` or set `LLAMACPP_SERVER_BIN`.
- `huggingface_hub` for download: `pip install huggingface_hub`.

## Usage

1. **Download models** (once):

   ```bash
   ./benchmark/next/download.sh
   ```

2. **Run all scenarios** (short only by default, quick):

   ```bash
   ./benchmark/next/benchmark.sh
   ```

3. **Run short + long context** (add long-context tests; takes longer):

   ```bash
   ./benchmark/next/benchmark.sh --long
   ```

4. **CPU pass is default** — each run does GPU then CPU (system-only, `N_GPU_LAYERS=0`) so results.txt has both sections. To skip the CPU pass: `RUN_CPU=0 ./benchmark/next/benchmark.sh`.

5. **Run one scenario manually**:

   ```bash
   ./benchmark/next/run_server.sh q2_full
   # In another terminal: curl or measure.py against http://localhost:18999
   ```

## Scenarios

| Scenario        | Model | MoE offload | KV cache-k | Use case              |
|----------------|-------|-------------|------------|------------------------|
| q2_full        | Q2    | no         | default    | Baseline (full GPU)    |
| q2_cache_k     | Q2    | no         | q4_1       | KV quant only         |
| q3_moe         | Q3    | yes        | default    | MoE penalty           |
| q3_moe_cache_k | Q3    | yes        | q4_1       | MoE + KV quant        |
| q4_moe         | Q4    | yes        | default    | Heavier quant + MoE   |
| q4_moe_cache_k | Q4    | yes        | q4_1       | Q4 + MoE + KV quant   |

Additional scenarios **q2_25gpu**, **q2_50gpu**, **q3_moe_50gpu**, **q4_moe_50gpu** use a GPU/CPU layer split (e.g. 50% of layers on GPU, rest on CPU). Adjust or add rows in scenarios.cfg with the 5th column set to an integer or `25%` / `50%` / `75%`.

Port **18999** is used so the benchmark does not clash with a normal LLM on 8000.

## Why 75% GPU can be faster than 100% GPU (and the pattern across models)

llama.cpp assigns layers to GPU by **keeping the last N layers on GPU**, not the first N. From `llama-model.cpp`:

```c
i_gpu_start = max(n_layer + 1 - n_gpu_layers, 0);
// Layer il is on GPU iff: il >= i_gpu_start && (il - i_gpu_start) < act_gpu_layers
```

So with **n_gpu_layers = 60** (75% of 80): **first 21 layers → CPU, last 60 layers → GPU.**  
So “75% GPU” means: **early layers (near input/embedding) on CPU, deep layers on GPU.**

That explains the pattern:

- **Early layers** are often more memory-bandwidth bound (embedding, first projections). Putting them on CPU **frees VRAM and GPU memory bandwidth** for the rest.
- **Deep layers** (the last 60) then run on GPU with **more headroom** — less contention, possibly better utilization.
- With **100% on GPU**, the whole model competes for VRAM and bandwidth, so the GPU can be slightly **more** bandwidth-limited and end up a bit slower than the 75% split.

So 75% GPU being at or above 100% GPU across Q2, Q3, and Q4 is **structural** (how the split is defined + memory bandwidth), not random variance. The same logic applies when comparing 25% / 50% / 75%: the “last N% on GPU” assignment determines which part runs where and why 75% is often the sweet spot.

## Comparison with other benchmarks

Below are **ballpark** references (different tools, models, and workloads). Our benchmark measures **chat completion tok/s** (real API, short prompt, 128 max_tokens); others often use llama-bench (synthetic) or different context.

| Source | Hardware | Model / setup | Token gen (tok/s) |
|--------|----------|----------------|-------------------|
| **Your run** (this benchmark) | RTX 5090 + 64GB | Qwen3-Coder-Next 80B-A3B, Q2 full GPU | **71** (q2_full), **64** (q2_cache_k) |
| **Your run** | RTX 5090 + 64GB | Qwen3-Coder-Next 80B-A3B, Q3/Q4 MoE offload | **25–27** (q3/q4_moe) |
| [Hardware Corner](https://www.hardware-corner.net/rtx-5090-llm-benchmarks/) | RTX 5090 32GB | Qwen3moe **30B**.A3B, full GPU, Q4_K_XL, llama-bench | **110–234** (4K–32K ctx) |
| [Hardware Corner](https://www.hardware-corner.net/rtx-5090-llm-benchmarks/) | RTX 5090 32GB | Qwen3 **32B** dense, full GPU, Q4_K_XL | **44–61** |
| [Hardware Corner](https://www.hardware-corner.net/guides/gpt-oss-offloading-moe-layers/) | RTX 5090 | **GPT-OSS 120B** MoE, MoE offload to CPU | **~8–9.6** |
| Community (L40S) | L40S 48GB | Qwen3-Next-80B-A3B GGUF | **42–89** (generation, various quants) |

**Takeaways**

- **q2_full ~71 tok/s** is in the same range as L40S reports for 80B-A3B (42–89) and is reasonable for an 80B model on a single GPU; Hardware Corner’s 30B MoE is faster (110–234) because it’s a smaller model (30B vs 80B).
- **q3/q4_moe ~25–27 tok/s** with MoE offload is much higher than GPT-OSS 120B with MoE offload (~8–9.6 tok/s), which fits with a smaller active model (80B vs 120B) and heavier quantization.
- Methodology differs: we use real `/v1/chat/completions`; many published numbers use llama-bench (fixed prompt length, no chat template). So treat these as relative comparisons, not exact matches.
