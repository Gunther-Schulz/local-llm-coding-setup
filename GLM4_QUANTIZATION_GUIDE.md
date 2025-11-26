# GLM-4-32B-0414 Quantization Guide

## ✅ Recommended Version: AWQ (4-bit)

**Use this repository:**
```
AMead10/GLM-4-32B-0414-awq
```

**Direct Link:**
https://huggingface.co/AMead10/GLM-4-32B-0414-awq

---

## 📊 Quantization Details

### AWQ (Activation-aware Weight Quantization) - ✅ RECOMMENDED

**Format:** W4A16 (Weights: 4-bit, Activations: 16-bit)

**Why AWQ for vLLM:**
- ✅ **Native vLLM support** - Works directly with `--quantization awq`
- ✅ **Best performance** - Optimized for inference speed
- ✅ **Pre-quantized** - Ready to use, no quantization needed
- ✅ **VRAM efficient** - ~20GB model weights (vs ~32GB FP16)

**VRAM Usage:**
- Model weights: ~20GB (4-bit AWQ)
- KV Cache (32K): ~2-3GB
- KV Cache (64K): ~4-6GB
- Activations: ~2-4GB
- **Total (64K context)**: ~26-30GB ✅ (fits in 32GB VRAM)

---

## 🔍 Other Quantization Formats (Not Recommended for vLLM)

### GPTQ (4-bit)
- **Format:** W4A16 (similar to AWQ)
- **Compatibility:** ✅ Works with vLLM (`--quantization gptq`)
- **Availability:** May exist but AWQ is preferred
- **Why not:** AWQ generally performs better for inference

### GGUF (Various bit levels)
- **Formats:** Q4_K_S, Q4_K_M, Q6_K, Q8_0, etc.
- **Compatibility:** ❌ **NOT compatible with vLLM**
- **Use with:** llama.cpp, Ollama, or other GGUF-compatible frameworks
- **Why not:** vLLM doesn't support GGUF format

### DWQ (Dynamic Weight Quantization)
- **Format:** 4-bit
- **Compatibility:** ⚠️ May not be directly supported by vLLM
- **Why not:** AWQ is the standard for vLLM

---

## 📥 Download Command

**Recommended (AWQ):**
```bash
huggingface-cli download AMead10/GLM-4-32B-0414-awq \
  --local-dir /workspace/models/glm-4-32b-0414-awq
```

**Alternative (if you find GPTQ version):**
```bash
# Search for GPTQ version first:
# https://huggingface.co/models?search=GLM-4-32B-0414+gptq

# If found:
huggingface-cli download <found-gptq-repo> \
  --local-dir /workspace/models/glm-4-32b-0414-gptq
```

---

## 🚀 vLLM Startup Commands

### With AWQ (Recommended)

**32K context (native, safe):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm-4-32b-0414-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1
```

**64K context (with YaRN, recommended):**
```bash
# First configure YaRN in config.json (see YARN_EXTENDED_CONTEXT_SETUP.md)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm-4-32b-0414-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1
```

### With GPTQ (If AWQ not available)

```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm-4-32b-0414-gptq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --quantization gptq \
  --tensor-parallel-size 1
```

---

## 📋 Comparison Table

| Format | Bit Depth | vLLM Compatible | VRAM (Model) | Performance | Recommendation |
|--------|-----------|-----------------|-------------|-------------|----------------|
| **AWQ** | 4-bit | ✅ Yes | ~20GB | ⭐⭐⭐⭐⭐ Best | ✅ **USE THIS** |
| **GPTQ** | 4-bit | ✅ Yes | ~20GB | ⭐⭐⭐⭐ Very Good | ✅ Alternative |
| **FP16** | 16-bit | ✅ Yes | ~32GB | ⭐⭐⭐⭐⭐ Best Quality | ⚠️ Too large for 32GB |
| **GGUF** | Various | ❌ No | Varies | N/A | ❌ Use llama.cpp instead |

---

## ✅ Final Recommendation

**Use: `AMead10/GLM-4-32B-0414-awq`**

**Why:**
1. ✅ **Pre-quantized** - Ready to use immediately
2. ✅ **4-bit AWQ** - Optimal balance of quality and VRAM usage
3. ✅ **vLLM compatible** - Works with `--quantization awq` flag
4. ✅ **Proven** - Used by many in the community
5. ✅ **Fits in 32GB VRAM** - With room for extended context (64K)

**Download:**
```bash
huggingface-cli download AMead10/GLM-4-32B-0414-awq \
  --local-dir /workspace/models/glm-4-32b-0414-awq
```

**Start vLLM:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm-4-32b-0414-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --quantization awq
```

---

## 🔗 Useful Links

- **AWQ Repository**: https://huggingface.co/AMead10/GLM-4-32B-0414-awq
- **Search for GPTQ**: https://huggingface.co/models?search=GLM-4-32B-0414+gptq
- **vLLM AWQ Docs**: https://docs.vllm.ai/en/latest/models/quantization/awq.html

---

## 📝 Notes

- **AWQ is 4-bit by default** - This is the standard quantization level
- **No 8-bit AWQ needed** - 4-bit provides excellent quality with significant VRAM savings
- **GGUF is not an option** - vLLM doesn't support GGUF format
- **GPTQ is acceptable** - But AWQ is generally preferred for vLLM


