# Quantized Models Compatible with vLLM

## ⚠️ Important Finding: No Official Qwen2.5-Coder-16B Model

**The Qwen2.5-Coder series only has these official sizes:**
- ✅ **7B**: Qwen2.5-Coder-7B-Instruct
- ✅ **32B**: Qwen2.5-Coder-32B-Instruct  
- ❌ **16B**: Does NOT exist officially

**However, there is a 16B coding model available:**
- ✅ **DeepSeek Coder V2 16B**: DeepSeek-Coder-V2-Lite-Instruct (16B parameters)

---

## 🔍 Available Quantized Models for vLLM

### vLLM Compatibility Requirements
- **AWQ (Activation-aware Weight Quantization)**: ✅ Fully supported
- **GPTQ**: ✅ Supported (with `--quantization gptq`)
- **GGUF**: ❌ NOT supported (use llama.cpp/Ollama instead)
- **FP16/BF16**: ✅ Supported (no quantization flag needed)

---

## 📋 Recommended Models (16B Size Range)

### Option 1: DeepSeek Coder V2 16B (AWQ) ⭐ RECOMMENDED

**Model Information:**
- **Repository**: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- **Size**: 16B parameters
- **Context**: 128K tokens (native)
- **VRAM Usage (AWQ)**: ~20-24GB
- **VRAM Usage (FP16)**: ~26-30GB

**Quantized Versions:**
- **AWQ**: Check for community-quantized versions or quantize yourself
- **FP16**: Available directly from repository

**Download Command:**
```bash
# FP16 version (direct from Hugging Face)
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
  --local-dir /workspace/models/deepseek-coder-v2-16b

# Check for AWQ version (search Hugging Face)
# Look for: deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct-AWQ
# Or community versions like: *-awq or *-gptq
```

**vLLM Startup Command:**
```bash
# With AWQ (if available)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/deepseek-coder-v2-16b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --quantization awq \
  --tensor-parallel-size 1

# With FP16 (if AWQ not available)
python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/deepseek-coder-v2-16b \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 131072 \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size 1
```

---

### Option 2: Qwen2.5-Coder-7B (AWQ) - Smaller Alternative

**Model Information:**
- **Repository**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- **Size**: 7B parameters
- **Context**: 128K tokens
- **VRAM Usage (AWQ)**: ~4-6GB
- **VRAM Usage (FP16)**: ~7-9GB

**Quantized Versions:**
- **AWQ**: Check for community-quantized versions
- **FP16**: Available directly

**Download Command:**
```bash
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct \
  --local-dir /workspace/models/qwen2.5-coder-7b
```

---

### Option 3: Qwen2.5-Coder-32B (AWQ) - Larger Alternative

**Model Information:**
- **Repository**: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
- **Size**: 32B parameters
- **Context**: 64K tokens (native, can extend to 128K with YaRN)
- **VRAM Usage (AWQ)**: ~26-30GB
- **VRAM Usage (FP16)**: ~32GB+ (may not fit)

**Quantized Versions:**
- **AWQ**: Available (check for pre-quantized or quantize yourself)
- **FP16**: Available directly

**Download Command:**
```bash
# Check for AWQ version first
# Look for: Qwen/Qwen2.5-Coder-32B-Instruct-AWQ
# Or community versions

# FP16 version
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct \
  --local-dir /workspace/models/qwen2.5-coder-32b
```

---

## 🔎 How to Find AWQ/GPTQ Quantized Versions

### Method 1: Search Hugging Face

1. **Search for AWQ versions:**
   ```
   https://huggingface.co/models?search=deepseek-coder-v2+awq
   https://huggingface.co/models?search=qwen2.5-coder+awq
   ```

2. **Search for GPTQ versions:**
   ```
   https://huggingface.co/models?search=deepseek-coder-v2+gptq
   https://huggingface.co/models?search=qwen2.5-coder+gptq
   ```

3. **Check model repository "Files" tab:**
   - Go to the model repository
   - Click "Files and versions"
   - Look for files with `-awq` or `-gptq` in the name
   - Or check for separate AWQ/GPTQ repositories

### Method 2: Community Quantized Models

**Common naming patterns:**
- `{model-name}-AWQ`
- `{model-name}-awq`
- `{model-name}-GPTQ`
- `{model-name}-gptq`
- `{author}/{model-name}-awq`
- `{author}/{model-name}-gptq`

**Example searches:**
- `TheBloke/DeepSeek-Coder-V2-Lite-Instruct-AWQ`
- `TheBloke/Qwen2.5-Coder-32B-Instruct-AWQ`
- `casperhansen/DeepSeek-Coder-V2-Lite-Instruct-AWQ`

### Method 3: Check Model Cards

Many model repositories list quantized versions in their README or model card. Look for:
- "Quantized versions" section
- Links to AWQ/GPTQ variants
- Community contributions

---

## 🛠️ Quantizing Models Yourself (If Needed)

If no pre-quantized version exists, you can quantize models yourself:

### AWQ Quantization

```bash
# Install AutoAWQ
pip install autoawq

# Quantize model
python -m awq.entrypoint.quantize \
  --model_path /path/to/original/model \
  --output_path /path/to/quantized/model \
  --w_bit 4 \
  --q_group_size 128
```

### GPTQ Quantization

```bash
# Install AutoGPTQ
pip install auto-gptq

# Quantize model
python -m auto_gptq.entrypoint.quantize \
  --model_path /path/to/original/model \
  --output_path /path/to/quantized/model \
  --bits 4 \
  --group_size 128
```

**Note**: Quantization requires significant VRAM and time. Pre-quantized models are preferred.

---

## 📊 Model Comparison Table

| Model | Size | Context | VRAM (AWQ) | VRAM (FP16) | vLLM Compatible | Download Link |
|-------|------|---------|------------|-------------|------------------|---------------|
| **DeepSeek Coder V2** | 16B | 128K | ~20-24GB | ~26-30GB | ✅ | [Link](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct) |
| **Qwen2.5-Coder-7B** | 7B | 128K | ~4-6GB | ~7-9GB | ✅ | [Link](https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct) |
| **Qwen2.5-Coder-32B** | 32B | 64K | ~26-30GB | ~32GB+ | ✅ | [Link](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct) |

---

## ✅ Recommended Action Plan

1. **For 16B size requirement:**
   - ✅ Use **DeepSeek Coder V2 16B** (closest to your requirement)
   - Search Hugging Face for AWQ/GPTQ versions
   - If not found, use FP16 (fits in 32GB VRAM)

2. **Verify quantized version exists:**
   ```bash
   # Search Hugging Face
   # Check model repository files
   # Look for community quantized versions
   ```

3. **Download and test:**
   ```bash
   # Download model
   huggingface-cli download <model-name> --local-dir /workspace/models/<model-dir>
   
   # Test with vLLM
   python -m vllm.entrypoints.openai.api_server \
     --model /workspace/models/<model-dir> \
     --quantization awq  # or gptq, or omit for FP16
   ```

---

## 🔗 Useful Links

- **Hugging Face Model Search**: https://huggingface.co/models
- **DeepSeek Coder V2**: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- **Qwen2.5-Coder-7B**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct
- **Qwen2.5-Coder-32B**: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
- **vLLM Documentation**: https://docs.vllm.ai/
- **AWQ Documentation**: https://github.com/mit-han-lab/llm-awq
- **GPTQ Documentation**: https://github.com/AutoGPTQ/AutoGPTQ

---

## 📝 Next Steps

1. **Decide on model**: DeepSeek Coder V2 16B (recommended for 16B size)
2. **Search for quantized version**: Check Hugging Face for AWQ/GPTQ versions
3. **If found**: Download and use with `--quantization awq` or `--quantization gptq`
4. **If not found**: Use FP16 version (still fits in 32GB VRAM)
5. **Proceed with RunPod setup**: Use the selected model in your setup scripts


