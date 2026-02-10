# Qwen3-Coder-Next MoE / KV-cache benchmark

Standalone llama.cpp testing (no proxy): compare Qwen3-Coder-Next 80B (Q2/Q3/Q4, MoE offload), Qwen3-Coder-30B-A3B, and GLM-4.7-Flash on your 5090. Port **18999**.

## Definitely try first (single RTX 5090 / 32GB, coding)

| Priority | Scenario | Why | Command to test |
|----------|----------|-----|------------------|
| **1** | **glm47_mxfp4_full** | Best full-GPU coding on 5090: 59.2% SWE-bench, ~17 GB, 131K context, ~120–158 tok/s | `./benchmark/next/run_server.sh glm47_mxfp4_full 18999` then `python3 benchmark/next/measure.py --port 18999 --model glm-4.7-flash` |
| **2** | **qwen30b_q4xl_full** | Highest Qwen coding quality that fits full GPU: 30B Q4_K_XL, ~17 GB, 147K context, ~110–234 tok/s | `./benchmark/next/run_server.sh qwen30b_q4xl_full 18999` then `python3 benchmark/next/measure.py --port 18999 --model qwen3-coder-30b-a3b` |
| **3** | **q4m_moe** or **q4_moe** | Highest coding quality that fits (with offload): Coder-Next 80B Q4, experts on CPU, ~25–30 tok/s | `./benchmark/next/run_server.sh q4m_moe 18999` then `python3 benchmark/next/measure.py --port 18999` |
| **4** | **q2_full** | Coder-Next 80B full GPU baseline: fits 32 GB, ~70 tok/s | `./benchmark/next/run_server.sh q2_full 18999` then `python3 benchmark/next/measure.py --port 18999` |
| **5** | **iq3xxs_full** | Coder-Next 80B, better than Q2, still full GPU (~31 GB) | `./benchmark/next/run_server.sh iq3xxs_full 18999` then `python3 benchmark/next/measure.py --port 18999` |

Run the server in one terminal; in another, run `measure.py` or point your client at `http://127.0.0.1:18999/v1`.

## Models covered

| Model | Scenarios | Notes |
|-------|-----------|--------|
| **GLM-4.7-Flash** 30B MoE | glm47_mxfp4_full | MXFP4_MOE, ~17 GB, full GPU, 131K context |
| **Qwen3-Coder-30B-A3B** | qwen30b_q4xl_full | UD-Q4_K_XL, ~17 GB, full GPU, 147K context |
| **Qwen3-Coder-Next** 80B MoE | q2_full, q2_cache_k, q2_*gpu, iq3xxs_full, q3s_full, q3_moe*, q4m_full, q4m_moe, mxfp4_full, q4_moe* | Q2/Q3/Q4/IQ3/MXFP4; full GPU or MoE offload |

## Layout

- **download.sh** – Download all GGUF (Coder-Next, Coder-30B Q4_K_XL, GLM-4.7-Flash MXFP4_MOE). Run once.
- **scenarios.cfg** – Scenario definitions: `name|model_path|moe_ot|cache_k|n_gpu_layers[|api_model]`. Optional **api_model** for non-Qwen backends. **n_gpu_layers**: empty = all GPU; `0` = all CPU; integer or `25%`/`50%`/`75%` = GPU/CPU split (BENCHMARK_N_LAYERS=80 for Coder-Next).
- **run_server.sh** – Start llama-server for one scenario (port 18999). Context size defaults to 32K. Override via **--ctx** (e.g. `--ctx 128k`) or env **BENCHMARK_CTX=131072**. This sets the server’s `-c` (max context for the run); the model’s native max (e.g. 256K) is unchanged.
- **fill_context.sh** – Build long prompt for long-context tests.
- **measure.py** – Call `/v1/chat/completions`, report tokens and tok/s; with long prompt uses streaming and reports **gen_tok_s** (generation/decode speed). Use `--model` for non-default backends.
- **benchmark.py** – Main benchmark logic: start server → measure → stop; writes **results.txt**. Short column = tok/s; Long column = **gen/s** (decode speed). After each scenario prints **memory stats** (VRAM vs RAM from llama-server’s exit breakdown). Options: `--long`, `--short-only`, `--ctx 128k`, `--no-cpu`, and optional scenario name.
- **benchmark.sh** – Thin wrapper that runs **benchmark.py** (so you can use a conda env: `CONDA_BENCHMARK_ENV=myenv ./benchmark/next/benchmark.sh --long`).
- **parse_memory_breakdown.py** – Parses server log for llama.cpp’s memory breakdown table (GPU/Host model, context, compute in MiB). Used by benchmark.sh. **Note:** llama.cpp does not expose how many bytes were moved between RAM and VRAM during inference (e.g. when the model doesn’t fully fit in VRAM and parts are swapped); you only see the final allocation split.

## Prereqs

- llama-server (CUDA): `./external/llama.cpp/build-cuda/bin/llama-server` or set `LLAMACPP_SERVER_BIN`.
- `huggingface_hub` (or `huggingface-cli`) for download: `pip install huggingface_hub`.

## Usage

1. **Download all models** (once):

   ```bash
   ./benchmark/next/download.sh
   ```

2. **Run all scenarios** (short only by default):

   ```bash
   ./benchmark/next/benchmark.sh
   ```

3. **Run short + long context**:

   ```bash
   ./benchmark/next/benchmark.sh --long
   ```

4. **128K context** (server uses 128K of the model’s max, e.g. 256K):

   ```bash
   ./benchmark/next/benchmark.sh --ctx 128k mxfp4_full --long
   ```
   Also: `--ctx 131072` or `BENCHMARK_CTX=131072` (env).

5. **Skip CPU pass**: `./benchmark/next/benchmark.sh --no-cpu` or `RUN_CPU=0 ./benchmark/next/benchmark.sh`

6. **Run one scenario and measure** (example: GLM-4.7-Flash):

   ```bash
   ./benchmark/next/run_server.sh glm47_mxfp4_full 18999
   # In another terminal:
   python3 benchmark/next/measure.py --port 18999 --model glm-4.7-flash
   ```

   For Qwen3-Coder-Next scenarios (q2_full, q4m_moe, etc.) you can omit `--model` (default is `qwen3-coder-next`). For Qwen3-Coder-30B use `--model qwen3-coder-30b-a3b`.

## Scenarios (summary)

| Scenario | Model | MoE offload | Use case |
|---------|-------|-------------|----------|
| glm47_mxfp4_full | GLM-4.7-Flash MXFP4_MOE | no | Best full-GPU coding (5090) |
| qwen30b_q4xl_full | Qwen3-Coder-30B Q4_K_XL | no | Highest Qwen full-GPU coding |
| q2_full, q2_cache_k, q2_*gpu | Coder-Next Q2 | no | Baseline, full GPU |
| iq3xxs_full | Coder-Next UD-IQ3_XXS | no | Better than Q2, full GPU |
| q3s_full | Coder-Next Q3_K_S | no | May fit 32GB |
| q3_moe*, q4m_moe, q4_moe* | Coder-Next Q3/Q4 | yes | Highest quality (offload), ~25–30 tok/s |
| mxfp4_full, q4m_full | Coder-Next MXFP4/Q4_K_M | no | Try full GPU (may OOM on 32GB) |

Full list and format in **scenarios.cfg**. Port **18999**.

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
