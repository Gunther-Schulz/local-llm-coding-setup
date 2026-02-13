# Benchmark (config/models)

Standalone llama.cpp testing (no proxy). **Scenarios use config/models:** each scenario is a model key (e.g. `qwen3-coder-next-mxfp4`); `run_server.sh` loads `config/models/<model_key>.yaml` (same as main `run_server.sh`). Port **18999**.

## Quick start

1. Ensure models are in `models/<model_key>/` (per `config/models/<model_key>.yaml` `gguf:`). Use main app: `./scripts/download-models.sh` (or specific keys: `./scripts/download-models.sh qwen3-coder-next-mxfp4`).
2. Run one scenario (scenario name = model key from scenarios.cfg):

   ```bash
   ./benchmark/next/run_server.sh qwen3-coder-next-mxfp4 18999
   # In another terminal:
   python3 benchmark/next/measure.py --port 18999 --model qwen3-coder-next-mxfp4
   ```

3. Run all scenarios: `./benchmark/next/benchmark.sh` (short only) or `./benchmark/next/benchmark.sh --long`.

## Scenarios

Scenarios are defined in **scenarios.cfg** and mirror **config/models/*.yaml**: one scenario per model key. Add a line to scenarios.cfg for any new model you add under config/models. Current scenarios include: glm-4.7-flash-q8-0, qwen3-coder-next-mxfp4, qwen3-coder-next-q8, qwen3-coder-next-bf16, qwen3-next-80b-abliterated-mxfp4, qwen3-next-80b-thinking-mxfp4, huihui-moe-4.8b-abliterated-mxfp4. To **compare two models in one run**, pass both scenario names: `./benchmark/next/benchmark.sh glm-4.7-flash-q8-0 qwen3-coder-next-mxfp4` (optionally add `--long`). Default fit (no n_gpu_layers override) is used so the server auto-fits to VRAM.

## Layout

- **scenarios.cfg** – Scenario list: `scenario_name|model_key|moe_ot|cache_k|n_gpu_layers[|api_model]`. **model_key** = `config/models/<model_key>.yaml` (same as ACTIVE_MODEL in main stack). n_gpu_layers: empty = use YAML; 0 = CPU; integer or 25%/50%/75% = override.
- **run_server.sh** – Loads config via `scripts/load_model_config.sh <model_key>`, starts llama-server (port 18999). Context/temp/top_p etc. from YAML; override with **BENCHMARK_CTX=131072** for 128K.
- **fill_context.sh** – Build long prompt for long-context tests.
- **measure.py** – Call `/v1/chat/completions`, report tokens and tok/s; with long prompt uses streaming and reports **gen_tok_s** (generation/decode speed). Use `--model` for non-default backends.
- **benchmark.py** – Main benchmark logic: start server → measure → stop; writes **results.txt**. Short column = tok/s; Long column = **gen/s** (decode speed). After each scenario prints **memory stats** (VRAM vs RAM from llama-server’s exit breakdown). Options: `--long`, `--short-only`, `--ctx 128k`, `--no-cpu`, and optional scenario name.
- **benchmark.sh** – Thin wrapper that runs **benchmark.py** (so you can use a conda env: `CONDA_BENCHMARK_ENV=myenv ./benchmark/next/benchmark.sh --long`).
- **parse_memory_breakdown.py** – Parses server log for llama.cpp’s memory breakdown table (GPU/Host model, context, compute in MiB). Used by benchmark.sh. **Note:** llama.cpp does not expose how many bytes were moved between RAM and VRAM during inference (e.g. when the model doesn’t fully fit in VRAM and parts are swapped); you only see the final allocation split.

## Prereqs

- llama-server (CUDA): `./external/llama.cpp/build-cuda/bin/llama-server` or set `LLAMACPP_SERVER_BIN`.
- For downloads: main app `./scripts/download-models.sh` (requires aria2, curl, python3, PyYAML).

## Usage

1. **Download models** (once, via main app): `./scripts/download-models.sh` (all from config/models) or `./scripts/download-models.sh <model_key> ...`.

2. **Run all scenarios** (short only by default):

   ```bash
   ./benchmark/next/benchmark.sh
   ```

3. **Run short + long context**:

   ```bash
   ./benchmark/next/benchmark.sh --long
   ```

4. **128K context**: `./benchmark/next/benchmark.sh --ctx 128k qwen3-coder-next-mxfp4 --long` or `BENCHMARK_CTX=131072`.

5. **Skip CPU pass**: `./benchmark/next/benchmark.sh --no-cpu` or `RUN_CPU=0 ./benchmark/next/benchmark.sh`

6. **Run one scenario and measure**: use scenario name (= model key) and same name for `--model`:

   ```bash
   ./benchmark/next/run_server.sh qwen3-coder-next-mxfp4 18999
   python3 benchmark/next/measure.py --port 18999 --model qwen3-coder-next-mxfp4
   ```

   Default `--model` is `qwen3-coder-next` if the scenario has no 6th column; otherwise use the api_model from scenarios.cfg (same as model_key).

Full list in **scenarios.cfg** (one scenario per config/models entry). Port **18999**.

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
| **Your run** | RTX 5090 32GB | Qwen3-Coder-Next 80B MXFP4_MOE (mxfp4_full) | **~39** short, **~51** long gen/s |
| **Your run** | RTX 5090 + 64GB | Qwen3-Coder-Next 80B-A3B, Q3/Q4 MoE offload | **25–27** (q3/q4_moe) |
| **Your run** | RTX 5090 32GB | GLM-4.7-Flash MXFP4_MOE, full GPU (glm47_mxfp4_full) | **~120–158** (4K–32K ctx) |
| **Your run** | RTX 5090 32GB | Qwen3-Coder-30B Q4_K_XL, full GPU (qwen30b_q4xl_full) | **~110–234** (4K–32K ctx) |
| [Hardware Corner](https://www.hardware-corner.net/rtx-5090-llm-benchmarks/) | RTX 5090 32GB | Qwen3moe **30B**.A3B, full GPU, Q4_K_XL, llama-bench | **110–234** (4K–32K ctx) |
| [Hardware Corner](https://www.hardware-corner.net/rtx-5090-llm-benchmarks/) | RTX 5090 32GB | Qwen3 **32B** dense, full GPU, Q4_K_XL | **44–61** |
| [Hardware Corner](https://www.hardware-corner.net/guides/gpt-oss-offloading-moe-layers/) | RTX 5090 | **GPT-OSS 120B** MoE, MoE offload to CPU | **~8–9.6** |
| Community (L40S) | L40S 48GB | Qwen3-Next-80B-A3B GGUF | **42–89** (generation, various quants) |

**Takeaways**

- **q2_full ~71 tok/s** is in the same range as L40S reports for 80B-A3B (42–89) and is reasonable for an 80B model on a single GPU; Hardware Corner’s 30B MoE is faster (110–234) because it’s a smaller model (30B vs 80B).
- **mxfp4_full ~39 short / ~51 long gen/s**: No published tok/s benchmarks found for Qwen3-Coder-Next 80B MXFP4_MOE; your numbers are a useful reference. Compared to this repo: lower than q2_full (71), higher than q3/q4_moe (25–27), in the L40S 80B range (42–89).
- **q3/q4_moe ~25–27 tok/s** with MoE offload is much higher than GPT-OSS 120B with MoE offload (~8–9.6 tok/s), which fits with a smaller active model (80B vs 120B) and heavier quantization.
- Methodology differs: we use real `/v1/chat/completions`; many published numbers use llama-bench (fixed prompt length, no chat template). So treat these as relative comparisons, not exact matches.
