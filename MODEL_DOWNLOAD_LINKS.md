# Direct Download Links for Quantized Models (vLLM Compatible)

## 🎯 Target: 16B Coding Model with Quantization

### ⚠️ Key Finding: No Qwen2.5-Coder-16B Exists

**Official Qwen2.5-Coder sizes:**
- 7B ✅
- 32B ✅
- 16B ❌ (does not exist)

**Solution: Use DeepSeek Coder V2 16B** (excellent coding model, 16B parameters)

---

## 📥 Recommended Model: DeepSeek Coder V2 16B

### Option A: FP16 (Full Precision) - Available Now

**Direct Download:**
```bash
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --local-dir /workspace/models/deepseek-coder-v2-16b
```

**Repository**: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

**VRAM Usage**: ~26-30GB (fits in 32GB VRAM)
**Context**: 128K tokens
**vLLM Command**: (no `--quantization` flag needed)

---

### Option B: AWQ Quantized (If Available)

**Search for AWQ versions:**
1. Visit: https://huggingface.co/models?search=deepseek-coder-v2+awq
2. Look for repositories like:
   - `TheBloke/DeepSeek-Coder-V2-Lite-Instruct-AWQ`
   - `casperhansen/DeepSeek-Coder-V2-Lite-Instruct-AWQ`
   - `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-AWQ`

**If found, download:**
```bash
huggingface-cli download <author>/DeepSeek-Coder-V2-Lite-Instruct-AWQ \
  --local-dir /workspace/models/deepseek-coder-v2-16b-awq
```

**VRAM Usage**: ~20-24GB (more headroom)
**vLLM Command**: Add `--quantization awq`

---

### Option C: GPTQ Quantized (If Available)

**Search for GPTQ versions:**
1. Visit: https://huggingface.co/models?search=deepseek-coder-v2+gptq
2. Look for repositories like:
   - `TheBloke/DeepSeek-Coder-V2-Lite-Instruct-GPTQ`
   - `deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-GPTQ`

**If found, download:**
```bash
huggingface-cli download <author>/DeepSeek-Coder-V2-Lite-Instruct-GPTQ \
  --local-dir /workspace/models/deepseek-coder-v2-16b-gptq
```

**VRAM Usage**: ~20-24GB (similar to AWQ)
**vLLM Command**: Add `--quantization gptq`

---

## 🔍 How to Verify Quantized Versions Exist

### Quick Check Script

```bash
#!/bin/bash
# Check for quantized versions

echo "Searching for AWQ versions..."
curl -s "https://huggingface.co/api/models?search=deepseek-coder-v2+awq" | grep -o '"id":"[^"]*"' | head -5

echo ""
echo "Searching for GPTQ versions..."
curl -s "https://huggingface.co/api/models?search=deepseek-coder-v2+gptq" | grep -o '"id":"[^"]*"' | head -5
```

### Manual Search Links

**AWQ Search:**
- https://huggingface.co/models?search=deepseek-coder-v2+awq
- https://huggingface.co/models?search=deepseek-coder-v2-lite+awq

**GPTQ Search:**
- https://huggingface.co/models?search=deepseek-coder-v2+gptq
- https://huggingface.co/models?search=deepseek-coder-v2-lite+gptq

---

## 📋 Complete Download & Setup Commands

### Step 1: Check for Quantized Versions

```bash
# Search Hugging Face website or use API
# Visit: https://huggingface.co/models?search=deepseek-coder-v2+awq
```

### Step 2: Download Model

**If AWQ found:**
```bash
huggingface-cli download <found-awq-repo> \
  --local-dir /workspace/models/deepseek-coder-v2-16b-awq
```

**If GPTQ found:**
```bash
huggingface-cli download <found-gptq-repo> \
  --local-dir /workspace/models/deepseek-coder-v2-16b-gptq
```

**If no quantized version (use FP16):**
```bash
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --local-dir /workspace/models/deepseek-coder-v2-16b
```

### Step 3: Start vLLM Server

**With AWQ:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/deepseek-coder-v2-16b-awq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1
```

**With GPTQ:**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/deepseek-coder-v2-16b-gptq \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --quantization gptq \
  --tensor-parallel-size 1
```

**With FP16 (no quantization):**
```bash
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/deepseek-coder-v2-16b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1
```

---

## 🎯 Recommended Action Plan

1. **First, search for quantized versions:**
   - Visit: https://huggingface.co/models?search=deepseek-coder-v2+awq
   - Check if any AWQ or GPTQ versions exist
   - Note the repository name if found

2. **If quantized version exists:**
   - Download the AWQ/GPTQ version
   - Use `--quantization awq` or `--quantization gptq` flag
   - Saves ~6-10GB VRAM

3. **If no quantized version:**
   - Download FP16 version (still fits in 32GB VRAM)
   - Use without `--quantization` flag
   - Works perfectly, just uses more VRAM

4. **Proceed with RunPod setup:**
   - Use the downloaded model path
   - Configure compression proxy
   - Test the setup

---

## 🔗 Direct Links to Check

### DeepSeek Coder V2 16B
- **Main Repository**: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- **Files Tab**: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/tree/main
- **AWQ Search**: https://huggingface.co/models?search=deepseek-coder-v2+awq
- **GPTQ Search**: https://huggingface.co/models?search=deepseek-coder-v2+gptq

### Alternative: Qwen2.5-Coder-7B (Smaller)
- **Main Repository**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- **AWQ Search**: https://huggingface.co/models?search=qwen2.5-coder-7b+awq

### Alternative: Qwen2.5-Coder-32B (Larger)
- **Main Repository**: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
- **AWQ Search**: https://huggingface.co/models?search=qwen2.5-coder-32b+awq

---

## ✅ Next Steps

1. **Visit the search links above** to check for quantized versions
2. **If found**: Note the repository name and use it in download command
3. **If not found**: Use FP16 version (works perfectly, just uses more VRAM)
4. **Proceed with RunPod setup** using the model you've downloaded

---

## 📝 Notes

- **FP16 is always available** - if no quantized version exists, FP16 works fine
- **32GB VRAM is sufficient** for DeepSeek Coder V2 16B in FP16 format
- **Quantization saves VRAM** but may slightly reduce quality
- **128K context** is available with DeepSeek Coder V2 16B (native support)


