# YaRN Extended Context Setup for RTX 5090 (32GB VRAM)

## ✅ Strategy: Large Quantized Model + YaRN Extended Context

**Your approach is correct!** Here's why it works:

1. **AWQ Quantization** reduces model weights from ~32GB (FP16) to ~20GB (4-bit AWQ)
2. **Saved VRAM** (~12GB) can be used for larger KV cache (extended context)
3. **YaRN** extends context beyond native limits
4. **Result**: Larger model quality + Extended context = Best of both worlds

---

## 📊 VRAM Breakdown: 32B AWQ Model with Extended Context

### Option 1: 64K Context (Recommended, Safe)

**Model**: Qwen2.5-Coder-32B (AWQ) or GLM-4-32B-0414 (AWQ)

**VRAM Usage:**
- Model weights (AWQ): ~20GB
- KV Cache (64K): ~4-6GB
- Activations: ~2-4GB
- **Total: ~26-30GB** ✅ (fits comfortably, 2-6GB headroom)

**Native vs Extended:**
- **Qwen2.5-Coder-32B**: Native 64K - YaRN can extend to 128K
- **GLM-4-32B-0414**: Native 32K - YaRN extends to 64K (recommended)

---

### Option 2: 128K Context (Aggressive, May Be Tight)

**Model**: Qwen2.5-Coder-32B (AWQ) with YaRN

**VRAM Usage:**
- Model weights (AWQ): ~20GB
- KV Cache (128K): ~8-10GB ⚠️ (4x larger than 32K)
- Activations: ~2-4GB
- **Total: ~30-34GB** ⚠️ (may exceed 32GB VRAM, tight fit)

**Recommendation**: 
- ✅ **64K context is safer** and still very large
- ⚠️ **128K context is possible** but may cause OOM errors
- 💡 **Test 128K** if you need it, but monitor VRAM usage

---

## 🎯 Recommended Models for YaRN Extension

### Option 1: Qwen2.5-Coder-32B (AWQ) ⭐ BEST FOR CODING

**Why:**
- ✅ Excellent coding model
- ✅ Native 64K context (no YaRN needed for 64K)
- ✅ Can extend to 128K with YaRN (if needed)
- ✅ Pre-quantized AWQ versions available

**Download:**
```bash
# Check for AWQ version first
# Search: https://huggingface.co/models?search=qwen2.5-coder-32b+awq

# If AWQ found:
huggingface-cli download <found-awq-repo> \
  --local-dir /workspace/models/qwen2.5-coder-32b-awq

# If not found, use FP16 (still works, just uses more VRAM):
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct \
  --local-dir /workspace/models/qwen2.5-coder-32b
```

**vLLM Startup (64K native - no YaRN needed):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/qwen2.5-coder-32b-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1
```

**vLLM Startup (128K with YaRN - experimental):**
```bash
# First, configure YaRN in model's config.json (see below)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/qwen2.5-coder-32b-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1
```

---

### Option 2: GLM-4-32B-0414 (AWQ) ⭐ BEST FOR REASONING

**Why:**
- ✅ Strong reasoning capabilities (GPT-4o level)
- ✅ Pre-quantized AWQ version available
- ✅ Native 32K, perfect candidate for YaRN extension to 64K
- ✅ Multi-step code analysis

**Download:**
```bash
# Pre-quantized AWQ version available!
huggingface-cli download AMead10/GLM-4-32B-0414-awq \
  --local-dir /workspace/models/glm-4-32b-0414-awq
```

**vLLM Startup (32K native):**
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

**vLLM Startup (64K with YaRN - recommended):**
```bash
# First, configure YaRN in model's config.json (see below)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm-4-32b-0414-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1
```

---

## ⚙️ Configuring YaRN in Model Config

### Step 1: Locate Model's config.json

```bash
# Find the config.json file
ls -la /workspace/models/<model-dir>/config.json
```

### Step 2: Edit config.json

**For GLM-4-32B-0414 (extending 32K → 64K):**
```json
{
  "rope_scaling": {
    "type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 32768
  }
}
```

**For Qwen2.5-Coder-32B (extending 64K → 128K):**
```json
{
  "rope_scaling": {
    "type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 65536
  }
}
```

### Step 3: Verify Configuration

```bash
# Check if config.json was updated correctly
cat /workspace/models/<model-dir>/config.json | grep -A 3 "rope_scaling"
```

---

## 🚀 Complete RunPod Setup Script

### Step 1: Download Model

```bash
# Option A: GLM-4-32B-0414 AWQ (pre-quantized, ready to use)
huggingface-cli download AMead10/GLM-4-32B-0414-awq \
  --local-dir /workspace/models/glm-4-32b-0414-awq

# Option B: Qwen2.5-Coder-32B (check for AWQ first)
# Search: https://huggingface.co/models?search=qwen2.5-coder-32b+awq
huggingface-cli download <found-awq-repo> \
  --local-dir /workspace/models/qwen2.5-coder-32b-awq
```

### Step 2: Configure YaRN (if extending context)

```bash
# For GLM-4: Extend 32K → 64K
MODEL_DIR="/workspace/models/glm-4-32b-0414-awq"
python3 << EOF
import json

with open("$MODEL_DIR/config.json", "r") as f:
    config = json.load(f)

config["rope_scaling"] = {
    "type": "yarn",
    "factor": 2.0,
    "original_max_position_embeddings": 32768
}

with open("$MODEL_DIR/config.json", "w") as f:
    json.dump(config, f, indent=2)

print("✅ YaRN configuration added to config.json")
EOF
```

### Step 3: Start vLLM Server

```bash
# Activate conda environment
source /workspace/miniconda3/etc/profile.d/conda.sh
conda activate vllm-test

# Start vLLM with extended context
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm-4-32b-0414-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1
```

### Step 4: Monitor VRAM Usage

```bash
# Watch VRAM usage in real-time
watch -n 1 nvidia-smi

# Or check once
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

---

## 📊 Context Size vs VRAM Usage

| Model | Quantization | Native Context | Extended Context (YaRN) | VRAM Usage | Status |
|-------|-------------|----------------|------------------------|------------|--------|
| **GLM-4-32B** | AWQ | 32K | 64K (2x) | ~26-30GB | ✅ Safe |
| **GLM-4-32B** | AWQ | 32K | 128K (4x) | ~30-34GB | ⚠️ Tight |
| **Qwen2.5-Coder-32B** | AWQ | 64K | 128K (2x) | ~30-34GB | ⚠️ Tight |
| **Qwen2.5-Coder-32B** | AWQ | 64K | Native (no YaRN) | ~26-30GB | ✅ Safe |

**Recommendations:**
- ✅ **64K context**: Fits comfortably, recommended
- ⚠️ **128K context**: Possible but tight, test carefully
- 💡 **Start with 64K**, extend to 128K only if needed

---

## 🔧 Troubleshooting

### OOM (Out of Memory) Errors

**If you get OOM errors with 128K context:**

1. **Reduce context size:**
   ```bash
   --max-model-len 65536  # Instead of 131072
   ```

2. **Reduce GPU memory utilization:**
   ```bash
   --gpu-memory-utilization 0.85  # Instead of 0.9
   ```

3. **Use smaller context:**
   - 64K is still very large (most codebases fit)
   - Compression proxy can help manage context

### YaRN Not Working

**If extended context doesn't work:**

1. **Verify config.json:**
   ```bash
   cat /workspace/models/<model-dir>/config.json | grep rope_scaling
   ```

2. **Check vLLM version:**
   ```bash
   python -c "import vllm; print(vllm.__version__)"
   # Should be 0.11.2+ for YaRN support
   ```

3. **Check model uses RoPE:**
   - Qwen2.5-Coder: ✅ Uses RoPE
   - GLM-4: ✅ Uses RoPE
   - DeepSeek Coder V2: ✅ Uses RoPE

---

## ✅ Recommended Setup for RTX 5090

### Best Balance: GLM-4-32B-0414 AWQ + 64K Context

**Why:**
- ✅ Pre-quantized AWQ (ready to use)
- ✅ 32B model (excellent quality)
- ✅ 64K context (very large, fits comfortably)
- ✅ ~26-30GB VRAM usage (safe headroom)

**Setup:**
```bash
# 1. Download model
huggingface-cli download AMead10/GLM-4-32B-0414-awq \
  --local-dir /workspace/models/glm-4-32b-0414-awq

# 2. Configure YaRN (32K → 64K)
# Edit config.json as shown above

# 3. Start vLLM
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/glm-4-32b-0414-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.9 \
  --quantization awq
```

---

## 🎯 Summary

**Your strategy is correct:**
- ✅ Use larger quantized model (32B AWQ) - saves VRAM
- ✅ Use YaRN to extend context - uses saved VRAM for KV cache
- ✅ Result: Large model + Extended context = Best performance

**Recommended:**
- **Model**: GLM-4-32B-0414 AWQ or Qwen2.5-Coder-32B AWQ
- **Context**: 64K (safe) or 128K (aggressive, test first)
- **VRAM**: ~26-30GB (64K) or ~30-34GB (128K)

**Next Steps:**
1. Download quantized 32B model
2. Configure YaRN in config.json
3. Start vLLM with extended context
4. Monitor VRAM usage
5. Adjust context size if needed


