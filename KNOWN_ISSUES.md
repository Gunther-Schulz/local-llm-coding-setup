# Known issues: GLM + llama.cpp

This document describes problems we see when running **GLM-4.x** (especially GLM-4.7-Flash) GGUF with **llama-server** (llama.cpp). Our GLM model configs in `config/models/` reference this file; we document workarounds or settings where applicable (e.g. repeat_penalty, cache-reuse).

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

