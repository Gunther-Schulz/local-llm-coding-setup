# Known issues: GLM + llama.cpp, vLLM + Qwen3 Coder / GLM

This document describes problems we see when running **GLM-4.x** (especially GLM-4.7-Flash) GGUF with **llama-server** (llama.cpp), and **vLLM** with Qwen3 Coder or GLM. Our model configs in `config/models/` reference this file; we document workarounds or settings where applicable.

---

## vLLM: Qwen3 MoE / Qwen3-30B-A3B memory (OOM)

**Symptom:** vLLM OOM when loading or running Qwen3-30B-A3B (or Qwen3-Coder-30B-A3B) on 32 GB GPU, even with `max-model-len=1024` and `max-num-seqs=1`. KV cache for this model (4 KV heads, GQA) should be ~6 GB at 65K context — so “KV taking >50%” is not plausible; something else is wrong.

**Findings (online research):**

1. **CUDA graph capture OOM (vLLM V1 engine)**  
   [vLLM #17462](https://github.com/vllm-project/vllm/issues/17462): OOM happens during **CUDA graph capture** (warm-up), not during KV allocation. Same model (e.g. Qwen3-32B) works with V0 engine; V1 + CUDA graphs exhaust VRAM.  
   **Workaround:** `--enforce-eager` disables CUDA graphs so capture is skipped; model loads and runs (at lower throughput, e.g. ~20 tok/s vs ~60 with SGLang).  
   In this repo: add `--enforce-eager` via `serve_extra` in the Qwen model YAML if you hit this (see config comment).

2. **Driver fix (multi-GPU)**  
   [vLLM #17469](https://github.com/vllm-project/vllm/issues/17469): One user’s Qwen3-30B-A3B OOM on 4×A10 (96 GB) was **resolved by updating NVIDIA driver 535 → 550**; same config then worked. Worth trying on multi-GPU before relying on enforce-eager.

3. **Hybrid / shared KV cache over-allocation (Ascend)**  
   [vllm-ascend #3368](https://github.com/vllm-project/vllm-ascend/issues/3368): For Qwen3-Next (hybrid full + linear attention), `initialize_kv_cache_tensors` **allocates KV per layer in a “shared” group instead of reusing one buffer**, so actual memory exceeds `--gpu-memory-utilization`. Fix is in progress (e.g. PR #3760). Mainline vLLM on NVIDIA may have analogous behavior for hybrid/MoE models (e.g. full KV for layers that could share or use smaller window).

4. **SGLang**  
   Same Qwen3-30B-A3B setup was reported to run fine with SGLang 0.4.6.post1 (~60 tok/s) where vLLM 0.8.5 OOM’d. Alternative if you need throughput without enforce-eager.

**In this repo:** Qwen3 Coder YAML documents `serve_extra` and optional `--enforce-eager`. We do not auto-enable it; enable only if you see OOM during load or CUDA graph warm-up.

---

## Qwen3 Coder (llama.cpp): repetition loop (“Let me explore / Let me check”)

**Symptom:** The model gets stuck in a loop repeating exploratory phrases like “Let me explore…”, “Let me first check…”, “I’ll help you understand the structure…”, “Let me examine…” without making progress.

**Cause:** Known repetition issue in code/exploration-style outputs: the model keeps starting new “Let me …” sentences instead of continuing. Qwen has reported loop issues ([Qwen3 #1220](https://github.com/QwenLM/Qwen3/issues/1220)); coding models also show “structural repetition” (same phrase pattern with minor variation). Greedy/beam decoding can reinforce this once it starts.

**Fix:**
1. **Increase `repeat_penalty`** in the model YAML. **Recommended online:** Qwen 2.5-Coder officially recommends **1.1** for repetition control; llama.cpp community suggests range **1.05–1.3** (don’t go above ~1.3 — see [llama.cpp #727](https://github.com/ggml-org/llama.cpp/issues/727); values like 1.9 can make loops worse). Unsloth’s default for this model is 1.05; raising to **1.1** or **1.15** is a documented, recommended adjustment.
2. **Optional:** Set **`repeat_last_n`** to ~128 (community: “don’t go above ~128”) so the penalty applies to a recent token window and can break phrase-level loops.
3. **Qwen llama.cpp docs:** For “repetition and endless generation” they also recommend **`--presence-penalty`** up to 2.0; our server script doesn’t pass it yet — add via env/flag if your build supports it.
4. **Prompt:** Ask for a single, direct answer to reduce “Let me explore…” step-by-step behavior.

**In this repo:** The Qwen3 Coder GGUF config uses `repeat_penalty: 1.1` (aligned with Qwen Coder’s recommended 1.1). If the loop persists, try 1.15. We do not pass `repeat_last_n` or `presence_penalty` from `run_server.sh`; add them via env/flag if your llama-server supports them.

---

## Qwen3 Coder (llama.cpp): optimal settings and still-current issues

**Sources:** [Unsloth Qwen3-Coder run guide](https://docs.unsloth.ai/models/qwen3-coder-how-to-run-locally), [Qwen llama.cpp docs](https://qwen.readthedocs.io/en/v3.0/run_locally/llama.cpp.html), unsloth `params` file, llama.cpp #727 / #331.

### Optimal settings (recommended online)

| Setting | Recommended | Our config | Notes |
|--------|-------------|------------|--------|
| **temp** | 0.7 | 0.7 | Qwen / Unsloth |
| **top_p** | 0.8 | 0.8 | Qwen / Unsloth |
| **top_k** | 20 | 20 | Qwen / Unsloth |
| **min_p** | 0.0 (0.01 OK) | 0.0 | **Set explicitly** — llama.cpp default is 0.1, which is wrong for Qwen |
| **repeat_penalty** | 1.05 (Unsloth) / 1.1 (Qwen Coder official) | 1.1 | We use 1.1 to reduce “Let me explore” loops |
| **context_size** | 65536 (or 32768 if OOM) | 65536 | Unsloth example uses 32768 for 30B; 65K is “adequate” per Qwen README |
| **batch_size / ubatch_size** | llama default 2048/512; higher can help throughput | 4096 / 4096 | Optional |
| **n_gpu_layers** | -1 (all on GPU) if VRAM fits | -1 | For 30B Q4_K_S ~17.5 GB on 32 GB GPU |
| **jinja** | true | true | Required for chat/tool format |
| **chat_template_file** | Custom template with “Do NOT omit the initial &lt;tool_call&gt; tag” | config/templates/Qwen3-Coder-tool-fix.jinja | See tool-calling issue below |
| **presence_penalty** | Up to 2.0 for “repetition and endless generation” (Qwen docs) | not passed | Add via env/flag if your llama-server supports it |
| **repeat_last_n** | ~128 (llama.cpp community) | not passed | Add via env/flag if needed for phrase-level loops |
| **cache_type_k / cache_type_v** | q8_0 or q4_1 for long context / less VRAM (Unsloth) | not set | Optional; needs Flash Attention build for V cache |

No other setting changes are required for “optimal” per current docs; the table above is the full set.

### Still-current issues

1. **Repetition loop (“Let me explore / Let me check”)** — See section above. Use `repeat_penalty: 1.1` (we do); optionally `presence_penalty` or `repeat_last_n` if your server supports them.

2. **Tool calling: 30B omits `<tool_call>` tag** ([Qwen3-Coder #475](https://github.com/QwenLM/Qwen3-Coder/issues/475), open)  
   The 30B model often omits the opening `<tool_call>` tag, especially when a tool call follows text. **Fix:** Use a chat template that explicitly says “Do NOT omit the initial &lt;tool_call&gt; tag” and stricter instructions. We use `config/templates/Qwen3-Coder-tool-fix.jinja`, which includes that. Tool calling can still be less reliable than other models (e.g. DeepSeek, Devstral).

3. **llama.cpp must support `qwen3moe`**  
   Qwen3/Coder GGUF requires llama.cpp build that supports the architecture (e.g. from version b5092). If you see “Architecture qwen3 not supported”, update llama.cpp and rebuild.

4. **min_p default**  
   llama.cpp default for `min_p` is 0.1; Qwen/Unsloth recommend **0.0**. Our config sets `min_p: 0.0` explicitly.

5. **Vulkan / LM Studio**  
   Some users report garbage output (e.g. “GGGG…”) or loops with Qwen3 GGUF on Vulkan backend in LM Studio. Prefer CUDA/Metal or another backend if you hit this.

6. **Chat template and tool arguments**  
   If the client sends tool `arguments` as a JSON string, the Jinja template can fail (“Can only get item pairs from a mapping”). Clients should pass parsed objects, or the template/server must handle string args (e.g. `json.loads`). vLLM/DashScope do this automatically; with llama-server, the client or proxy may need to.

7. **Long context**  
   For 65K–256K context, Unsloth recommends KV cache quantization (`--cache-type-k q8_0` or `q4_1`) to reduce VRAM and optionally `--flash-attn` with `--cache-type-v` (requires build with `GGML_CUDA_FA_ALL_QUANTS`). We don’t set these by default; add via env/flag if needed.

**In this repo:** The Qwen3 Coder GGUF YAML is aligned with the optimal settings above. We do not add `presence_penalty`, `repeat_last_n`, or `cache_type_*` from `run_server.sh`; add them via env or `serve_extra`/script if your build supports them.

### Relevance to our setup (build from `external/llama.cpp`, CUDA)

| Issue | Relevant to us? | Notes |
|--------|------------------|--------|
| **Repetition loop** | **Yes** | We use `repeat_penalty: 1.1`. Our `external/llama.cpp` server supports `--repeat-last-n` and `--presence-penalty` (CLI and API); we don’t pass them from `run_server.sh`. Clients can send them per request; or add env vars + script support to set server defaults. |
| **Tool calling 30B omits tag** | **Yes** | We use `config/templates/Qwen3-Coder-tool-fix.jinja` (“Do NOT omit the initial &lt;tool_call&gt; tag”). Mitigated; tool use can still be less reliable than other models. |
| **llama.cpp build (qwen3moe)** | **No** | Our `external/llama.cpp` has `LLM_ARCH_QWEN3MOE`, `qwen3moe.cpp`, and it’s in the build. As long as `build-cuda` is built from this tree, Qwen3 MoE is supported. |
| **min_p 0.0** | **Yes, done** | We set `min_p: 0.0` in the YAML and `run_server.sh` passes `--min-p` when `MIN_P` is set. |
| **Vulkan / LM Studio** | **No** | We use `build-cuda` (CUDA), not Vulkan. That issue doesn’t apply. |
| **Chat template / tool arguments** | **Only if** you use tool calling and the client sends tool `arguments` as a JSON string | Then Jinja can error (“Can only get item pairs from a mapping”). Most OpenAI-style clients send parsed objects. |
| **Long context** | **Optional** | We use `context_size: 65536`. `run_server.sh` already supports `CACHE_TYPE_K` and `CACHE_TYPE_V`; we don’t set them in the YAML. Add in YAML or env if you hit OOM or want longer context. See **KV cache (long context / OOM)** below. |

---

### KV cache (long context / OOM)

**When this matters:** You’re using **65K+ context** (e.g. `context_size: 65536` or higher) or you **see OOM** during generation. The KV cache holds Key and Value tensors for every layer and every token in context; at 65K tokens it can use several GB of VRAM. Quantizing the cache reduces that so you can fit longer context or avoid OOM.

**What the options do:**

| Option | Default | Effect |
|--------|--------|--------|
| **`--cache-type-k`** | f16 | Data type for the **Key** cache. Allowed: `f32`, `f16`, `bf16`, `q8_0`, `q4_0`, `q4_1`, `iq4_nl`, `q5_0`, `q5_1`. |
| **`--cache-type-v`** | f16 | Data type for the **Value** cache. Same allowed values. |

**Rough impact:** Going from f16 to **q8_0** halves the cache size for that part (K or V); **q4_0** / **q4_1** quarter it. So `cache_type_k: q8_0` (or `q4_1`) reduces **Key** cache VRAM; adding `cache_type_v: q8_0` (or `q4_1`) also reduces **Value** cache. That can free several GB at 65K context and help with OOM or allow longer context.

**Recommendation (Unsloth / community):** For long context (65K–256K), set **`cache_type_k`** to **q8_0** or **q4_1** (q4_1 a bit more accurate, slightly slower). Optionally set **`cache_type_v`** to the same if your build supports it; some backends require Flash Attention (`--flash-attn`) and a build with `GGML_CUDA_FA_ALL_QUANTS` for quantized V cache.

**In this repo:** We don’t set `cache_type_k` or `cache_type_v` in the Qwen3 Coder GGUF YAML. To use them:

1. **Env:** Export before starting the server, e.g.  
   `export CACHE_TYPE_K=q8_0`  
   and optionally `export CACHE_TYPE_V=q8_0`.  
   `run_server.sh` passes them as `--cache-type-k` and `--cache-type-v` when set.

2. **YAML:** In the model’s `llama:` block add:  
   `cache_type_k: q8_0`  
   and optionally `cache_type_v: q8_0`.  
   `load_model_config.sh` already emits `CACHE_TYPE_K` and `CACHE_TYPE_V` from the YAML.

**When to try it:** If you hit OOM at 65K context, or want to raise context (e.g. to 128K) without OOM, add `cache_type_k: q8_0` first; if you still need more headroom, add `cache_type_v: q8_0` (if your llama-server build supports it).

---

## vLLM: Qwen3 Coder tool parser

**Config:** Use `tool_call_parser: qwen3_coder` (not `qwen3`). vLLM registers `qwen3_coder` and `qwen3_xml`; there is no `qwen3` alias. Wrong name can break tool-call parsing when the client sends tools.

**Known upstream issues (vLLM + Qwen3-Coder):**
- **Tool call parsing:** [HF discussion](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/discussions/19): model outputs XML-style tool calls but vLLM may not parse them correctly (e.g. output like `<function=name><parameter=prompt>...` not picked up). Some users patch the parser or wait for upstream fix.
- **Streaming:** Parser may wait for full tool-argument value before returning, causing long delays for large params (e.g. file contents).
- **tool_choice:** Behavior can differ between `tool_choice="auto"` and `tool_choice="required"`; nested/object params may be double-encoded as strings in "auto" mode.
- **Security (CVE-2025-9141):** Older vLLM versions with `--tool-call-parser qwen3_coder` and `--enable-auto-tool-choice` used `eval()` on untrusted input in some code paths; upgrade vLLM if you use tool calling.

**In this repo:** We set `tool_call_parser: qwen3_coder` in Qwen3 Coder model YAMLs. We do **not** pass `--enable-auto-tool-choice` from `run_vllm.sh`; add it via `serve_extra` in the YAML if you need it (and ensure vLLM is patched for CVE above).

---

## vLLM: GLM-4.7-Flash (e.g. MXFP4)

**Config:** Use `tool_call_parser: glm47` (vLLM name for GLM 4.7 MoE). Model and tokenizer are set in the YAML (e.g. `zai-org/GLM-4.7-Flash` for tokenizer when using HF model id).

**Known issues:** MXFP4 models require `VLLM_MXFP4_USE_MARLIN=1` (our `run_vllm.sh` sets this when the model path contains `MXFP4`). If you see load or runtime errors, check vLLM and transformers versions (see `scripts/update-vllm-env.sh` for recommended versions).

---

## GLM + llama.cpp (sections below)

---

## 1. repeat_penalty must be disabled

**Symptom:** Poor or unstable output when using default repeat penalty with GLM-4.7-Flash.

**Cause:** GLM-4.7-Flash behaves better with repeat penalty turned off.

**Fix:** In the model YAML set `repeat_penalty: 1.0` (effectively disables it).

**In this repo:** All our GLM configs already set this.

---

## 2. Do not use --cache-reuse with GLM

**Symptom:** Crashes, corruption, or bad behavior when reusing KV cache across requests.

**Cause:** llama.cpp's `--cache-reuse` is not reliable with GLM models.

**Fix:** Do **not** pass `--cache-reuse` when serving GLM.

**In this repo:** `run_server.sh` does not add `--cache-reuse`. Our GLM YAMLs document this in a comment. If you run llama-server manually or via another script, omit this flag.

---

## 3. "Grammar still awaiting trigger" and tool-call corruption (llama.cpp #19068)

**Symptom:** Logs repeatedly show:

```text
Grammar still awaiting trigger after token N (`piece`)
```

- **Benign case:** The model is still in preamble, thinking, or plain text. Once it outputs `<tool_call>`, the tool-call grammar activates and parsing works. Occasional messages are normal.
- **Bug case:** The message repeats for thousands of tokens, output becomes gibberish, and RAM usage grows. The server has entered a bad state. Reported upstream as [llama.cpp issue #19068](https://github.com/ggml-org/llama.cpp/issues/19068). Seen with GLM-4.7-Flash when using tool definitions (OpenAI format via `--jinja`); the grammar trigger is not reliably applied.

**In this repo:** We have **not** fixed the root cause (the bug is in upstream llama.cpp). Our proxy does **not** implement any mitigation for this (e.g. no `tool_choice: "required"` or similar). If you hit the bug: force-kill and restart the server. Run without `--verbose` to reduce log noise from this message. **Investigation (high RAM vs vLLM):** see `docs/investigation-ram-grammar-vs-vllm.md`.

See upstream [llama.cpp #19068](https://github.com/ggml-org/llama.cpp/issues/19068). See also **config/templates/README.md** (section "Grammar still awaiting trigger") and **config/templates/GLM-4-tool.jinja** for template details.

---

## 4. High CPU when model is partly offloaded

**Not a bug.** When the model does not fully fit in VRAM, llama.cpp puts the remaining layers on the CPU and runs their full forward pass (attention + FFN) there. High CPU usage is the expected side effect.

**Original recommendation:** Someone recommended turning Flash Attention **globally off** (`flash_attn: off`) to go easier on the CPU when partly offloaded — but that loses FA on the GPU too.

**Our approach:** We rely on **selective Flash Attention** instead: FA on GPU layers, regular attention on CPU layers. So we still use FA on GPU (no loss there) and we do *not* use FA on CPU layers (easier on CPU for those layers). You get the CPU relief without turning FA off globally.

**Symptom:** Very high CPU usage when the GLM model does not fully fit in VRAM and some layers run on CPU.

**Cause:** Those layers run their compute on the CPU; the more layers (or threads) used for them, the higher the CPU load. This is how llama.cpp partial offload works, not a defect.

**In this repo:** We do not change offload behavior. We rely on llama.cpp's selective FA to avoid extra overhead: CPU-assigned layers use regular attention (no FA), GPU layers use FA. That way you don't need to set `flash_attn: off` globally and lose GPU performance.

**When is attention on CPU? (so selective FA matters)**

- **`n_gpu_layers: -1`** (our current configs): Request all layers on GPU. If VRAM doesn't fit, llama.cpp **overflows** only **FFN** (and MoE) to CPU — **attention stays on GPU**. So we never have attention on CPU; log says "all … on GPU (flash)". Selective FA is not needed in that case.
- **`n_gpu_layers` &lt; layer count** (e.g. `35` for a 48-layer model): Explicit partial offload — typical with **less VRAM or a larger model**. You set `n_gpu_layers` to a number so the model fits (e.g. 20 or 35). Then the layers that don't fit are **assigned** to CPU; their **attention** runs on CPU. Selective FA is **in effect**: FA on GPU layers, regular attention on those CPU layers. Log says "Flash Attention selective: N on GPU, M on CPU".

**Mitigation (optional):** In the model YAML you can try:

- `flash_attn: off`
- `cache_type_k: bf16`
- `cache_type_v: bf16`

Our GLM configs have these commented out; uncomment to experiment. They may reduce CPU load at the cost of some performance or compatibility. If the model fits fully in VRAM, you usually don't need these.

**Verification:** When Flash Attention is enabled, the server log reports whether it is "selective" (some layers on GPU, some on CPU) or "all on GPU". Look for `sched_reserve: Flash Attention selective:` or `Flash Attention: all … layers on GPU` to confirm.

---

## 5. "Does not natively describe tools" warning

**Symptom:** Server logs warn that the chat template supports tool calls but does not natively describe tools (so a generic fallback is used).

**Cause:** The default or embedded GLM template doesn't inject tool definitions in the format GLM expects.

**Fix:** Set in the model YAML:

```yaml
chat_template_file: config/templates/GLM-4-tool.jinja
```

That template injects tool definitions in GLM's format and removes the warning.

**In this repo:** Our GLM configs have this line commented; uncomment if you want native tool description.

---

## In this repo (changes we made)

- **Server logs:** Output is written directly to the log file (no tail wrapper), so logs update in real time and the server no longer appears "stuck" during startup.
- **Selective Flash Attention (§4):** FA on GPU layers, regular attention on CPU layers (so we don't turn FA off globally). In effect when `n_gpu_layers` &lt; layer count (e.g. less VRAM or larger model). With `n_gpu_layers: -1` (our configs) attention stays on GPU. Logs show selective vs all-GPU.

- **RAM/grammar issue (§3):** Documented only; not mitigated in this repo (our proxy has no such feature). Root cause remains in upstream llama.cpp (#19068).

- **32 GB max context (§6):** No reliable numbers; documented that users should tune `max_model_len` / `context_size` empirically.

