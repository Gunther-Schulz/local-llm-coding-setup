# DEFINITIVE Local LLM Setup for VS Code/Cursor Coding

> **Last Updated**: Based on discussion and research
> **Status**: Final choices and recommendations

---

## 🎯 Final Decisions

### Runtime
- **✅ vLLM** - Best performance, PagedAttention, OpenAI API compatible
- **Alternative: TensorRT-LLM** - NVIDIA's optimized framework (see comparison below)

### IDE & Extension
- **✅ VS Code** - Better local LLM support
- **✅ Continue Extension** - Best integration for local LLMs

### Hardware
- **✅ 32GB VRAM** - Available capacity

---

## 🤖 Model Selection

### Primary Recommendation: **Qwen2.5-Coder-32B (AWQ)**

**Why This Model:**
- ✅ **Best VRAM utilization**: Uses ~26-30GB (81-94% of 32GB)
- ✅ **Larger model = better quality**: 32B parameters vs 16B
- ✅ **64K context**: Sufficient for most coding tasks
- ✅ **Still leaves safety margin**: 2-6GB headroom

**Specifications:**
- **Model**: Qwen2.5-Coder-32B-Instruct
- **Format**: AWQ (4-bit quantization)
- **VRAM Usage**: ~26-30GB
- **Max Context**: 64K tokens
- **Download**: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct

**Memory Breakdown:**
- Model weights: ~20GB
- KV Cache (64K): ~4-6GB
- Activations: ~2-4GB
- **Total: ~26-30GB** ✅

---

### Alternative: **DeepSeek Coder V2 16B (AWQ)**

**When to Use:**
- Need 128K context for very large codebases
- Want to run other GPU applications simultaneously
- Prefer maximum context over maximum quality

**Specifications:**
- **Model**: DeepSeek-Coder-V2-Lite-Instruct
- **Format**: AWQ (4-bit quantization)
- **VRAM Usage**: ~20-24GB
- **Max Context**: 128K tokens
- **Download**: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct

**Memory Breakdown:**
- Model weights: ~10GB
- KV Cache (128K): ~8-10GB
- Activations: ~2-4GB
- **Total: ~20-24GB** ✅ (leaves 8-12GB unused)

---

### Alternative: **GLM-4-32B-0414 (AWQ)**

**When to Use:**
- Want a 32B model comparable to GPT-4o/DeepSeek-V3
- Need strong reasoning capabilities for complex coding tasks
- Prefer a model with extensive pre-training on reasoning-heavy data
- Want multi-step code reasoning and analysis

**Specifications:**
- **Model**: GLM-4-32B-0414
- **Format**: AWQ (4-bit quantization)
- **VRAM Usage**: ~24-27GB (32K context)
- **Max Context**: 32K tokens (extendable to 128K+ with YaRN)
- **Download**: https://huggingface.co/AMead10/GLM-4-32B-0414-awq

**What is YaRN?**
- **YaRN** (Yet another RoPE extensioN method) is a technique to extend context windows beyond the model's training length
- It modifies Rotary Position Embeddings (RoPE) to handle longer sequences
- Requires 10x fewer tokens and 2.5x fewer training steps than previous methods
- Can extend context from 32K to 128K+ tokens
- **Note**: Extending context with YaRN requires additional configuration in the model's `config.json`

**YaRN Compatibility with Other Models:**
- ✅ **Works with models using RoPE** (Rotary Position Embeddings)
- ✅ **Qwen2.5-Coder**: Uses RoPE - YaRN could theoretically extend beyond 64K (but already has 64K native)
- ✅ **DeepSeek Coder V2**: Uses RoPE - YaRN could theoretically extend beyond 128K (but already has 128K native)
- ⚠️ **Practical Note**: These models already have large native contexts, so YaRN extension may not be necessary
- ⚠️ **VRAM Impact**: Extending context with YaRN still increases VRAM usage (KV cache grows linearly)

**Why Consider:**
- ✅ **32B parameters**: Comparable to GPT-4o and DeepSeek-V3
- ✅ **Strong coding capabilities**: Excels at code generation, analysis, and function-call outputs
- ✅ **Multi-step reasoning**: Better at tracing logic and suggesting improvements
- ✅ **Pre-trained on 15T data**: Extensive reasoning-heavy training
- ✅ **vLLM compatible**: AWQ format works with your setup
- ⚠️ **Smaller context**: 32K vs 64K (Qwen) or 128K (DeepSeek)

**Memory Breakdown (AWQ, 32K context):**
- Model weights: ~20GB
- KV Cache (32K): ~2-3GB
- Activations: ~2-4GB
- **Total: ~24-27GB** ✅ (leaves 5-8GB headroom)

**Memory Breakdown (AWQ, 64K context with YaRN):**
- Model weights: ~20GB
- KV Cache (64K): ~4-6GB (2x larger than 32K)
- Activations: ~2-4GB
- **Total: ~26-30GB** ✅ (fits comfortably in 32GB VRAM)

**Memory Breakdown (AWQ, 128K context with YaRN):**
- Model weights: ~20GB
- KV Cache (128K): ~8-10GB ⚠️ (4x larger than 32K)
- Activations: ~2-4GB
- **Total: ~30-34GB** ⚠️ (may exceed 32GB VRAM)

**Important: Context Size and VRAM Usage**
- ✅ **Yes, more context uses MORE VRAM** - primarily due to KV cache
- KV cache grows **linearly** with context length (2x context = 2x KV cache memory)
- **32K → 64K**: KV cache doubles (~2-3GB → ~4-6GB)
- **32K → 128K**: KV cache quadruples (~2-3GB → ~8-10GB)
- **Recommendations**:
  - **32K context**: ~24-27GB VRAM ✅ Safe default
  - **64K context (with YaRN)**: ~26-30GB VRAM ✅ Fits comfortably, recommended if you need more context
  - **128K context (with YaRN)**: ~30-34GB VRAM ⚠️ May exceed 32GB, not recommended

**Note**: GLM-Z1-32B-0414 (reasoning variant) doesn't have a pre-quantized AWQ version. You'd need to quantize it yourself or use GGUF with Ollama/llama.cpp.

---

## 🧪 Testing Locally with 8GB VRAM (RTX 4060)

> **Note**: A complete `environment.yaml` file is provided in the repository root for easy conda environment setup with all dependencies.

### Quick Setup with Conda

**Create Conda Environment:**

**Option 1: Using environment.yaml (Recommended)**
```bash
# Create environment from yaml file
conda env create -f environment.yaml
conda activate vllm-test
```

**Option 2: Manual setup**
```bash
# Create conda environment with Python 3.10
conda create -n vllm-test python=3.10 -y
conda activate vllm-test

# Install vLLM and dependencies
pip install vllm huggingface-hub

# Optional: Install compression proxy dependencies (if using context compression)
pip install llmlingua sentence-transformers flask requests scikit-learn

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

### Small Models for Testing

If you want to test the setup locally before deploying to Runpod (32GB VRAM), you can use smaller models that fit in 8GB VRAM:

#### Option 1: **Qwen2.5-Coder-7B** (Recommended for Testing)

**Why This Model:**
- ✅ **Fits in 8GB VRAM**: ~7GB with FP16, ~4GB with AWQ
- ✅ **Coding-optimized**: Same family as your production model
- ✅ **Good for testing**: Validates the entire setup
- ✅ **Fast**: Quick inference for testing

**Specifications:**
- **Model**: Qwen2.5-Coder-7B-Instruct
- **Format**: FP16 or AWQ
- **VRAM Usage**: ~7GB (FP16) or ~4GB (AWQ)
- **Max Context**: 128K tokens (with FP16)
- **Download**: https://huggingface.co/Qwen/Qwen2.5-Coder-7B-Instruct

**Memory Breakdown (FP16, 8K context):**
- Model weights: ~7GB
- KV Cache (8K): ~1GB
- Activations: ~0.5GB
- **Total: ~8.5GB** ⚠️ Tight fit, use smaller context or AWQ

**Memory Breakdown (AWQ, 16K context):**
- Model weights: ~4GB
- KV Cache (16K): ~1GB
- Activations: ~0.5GB
- **Total: ~5.5GB** ✅ Comfortable

#### Option 2: **CodeLlama-7B-Instruct**

**Why This Model:**
- ✅ **Fits in 8GB VRAM**: ~7GB with FP16
- ✅ **Proven stable**: Well-tested coding model
- ✅ **Good for testing**: Validates setup

**Specifications:**
- **Model**: CodeLlama-7B-Instruct-hf
- **Format**: FP16
- **VRAM Usage**: ~7GB
- **Max Context**: 16K tokens
- **Download**: https://huggingface.co/codellama/CodeLlama-7b-Instruct-hf

### Complete Local Testing Setup (RTX 4060 - 8GB VRAM)

**Step 1: Create Conda Environment**
```bash
# Create and activate conda environment
conda create -n vllm-test python=3.10 -y
conda activate vllm-test

# Install vLLM and dependencies
pip install vllm huggingface-hub

# Verify CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

**Step 2: Download Test Model (Qwen2.5-Coder-7B)**
```bash
# Download model (will take a few minutes, ~14GB)
huggingface-cli download Qwen/Qwen2.5-Coder-7B-Instruct --local-dir ./models/qwen2.5-coder-7b
```

**Step 3: Start vLLM Server**
```bash
# For RTX 4060 (8GB VRAM) - Use 8K context to be safe
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-coder-7b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1
```

**Note**: Using `--gpu-memory-utilization 0.85` (85%) instead of 0.9 to leave more headroom on 8GB VRAM.

**Step 4: Test the Server**
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Test with a simple query
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "messages": [{"role": "user", "content": "Write a Python function to sort a list"}]
    }'
```

**Step 5: Configure Continue Extension**
Same as production setup - just point to `http://localhost:8000/v1`

### Alternative: CodeLlama-7B (If Qwen2.5-Coder-7B doesn't fit)

If you run into memory issues with Qwen2.5-Coder-7B:

```bash
# Download CodeLlama-7B
huggingface-cli download codellama/CodeLlama-7b-Instruct-hf --local-dir ./models/codellama-7b

# Start server with smaller context
python -m vllm.entrypoints.openai.api_server \
    --model ./models/codellama-7b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1
```

### RTX 4060 Specific Notes

- **8GB VRAM**: Use smaller context (4K-8K) to be safe
- **Memory utilization**: Use 0.85 instead of 0.9 for headroom
- **Monitor usage**: Run `nvidia-smi` to check actual VRAM usage
- **If OOM errors**: Reduce `--max-model-len` or `--gpu-memory-utilization`

### Testing Checklist

- [ ] Conda environment created and activated
- [ ] vLLM installed and verified
- [ ] CUDA available and GPU detected
- [ ] Small model downloaded (Qwen2.5-Coder-7B or CodeLlama-7B)
- [ ] vLLM server running on port 8000
- [ ] Server tested with curl
- [ ] Continue extension configured
- [ ] Tested with coding query in VS Code
- [ ] Verified compression proxy works (if using)
- [ ] Ready to scale up to production model on Runpod

### Quick Test Script

Create `test-local.sh`:
```bash
#!/bin/bash
# Activate conda environment
conda activate vllm-test

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-coder-7b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --tensor-parallel-size 1
```

Make executable: `chmod +x test-local.sh`
Run: `./test-local.sh`

### Notes for Testing

- **Use smaller context**: 8K-16K instead of 64K-128K
- **AWQ helps**: If available, use AWQ quantization for more headroom
- **Same setup**: The configuration process is identical to production
- **Validate workflow**: Test the entire pipeline before deploying

---

## 📋 Complete Setup

### Prerequisites

Before starting, ensure you have:

- **NVIDIA GPU** with 32GB VRAM (RTX 3090, A100, etc.)
- **CUDA 11.8+** installed and working
- **Python 3.10+** installed
- **Linux** (recommended) or Windows with WSL
- **VS Code** installed
- **~60-80GB free disk space** for models

**Verify CUDA:**
```bash
nvidia-smi  # Should show your GPU
nvcc --version  # Should show CUDA version
```

### 1. Install vLLM

**Option 1: Using Conda (Recommended)**
```bash
# Create environment from yaml file
conda env create -f environment.yaml
conda activate vllm-test

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

**Option 2: Using venv**
```bash
# Create virtual environment
python -m venv vllm-env
source vllm-env/bin/activate

# Install vLLM and dependencies
pip install vllm huggingface-hub

# Optional: Install compression proxy dependencies (if using context compression)
pip install llmlingua sentence-transformers flask requests scikit-learn

# Verify installation
python -c "import vllm; print(vllm.__version__)"
```

### 2. Download Model

**Primary Model (Qwen2.5-Coder-32B):**
```bash
pip install huggingface-hub
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir ./models/qwen2.5-coder-32b
```

**Alternative Model (DeepSeek Coder V2 16B):**
```bash
huggingface-cli download deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct --local-dir ./models/deepseek-coder-v2-16b
```

**Alternative Model (GLM-4-32B-0414 AWQ):**
```bash
huggingface-cli download AMead10/GLM-4-32B-0414-awq --local-dir ./models/glm-4-32b-0414-awq
```

### 3. Start vLLM Server

**For Qwen2.5-Coder-32B (Primary):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-coder-32b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1
```

**For DeepSeek Coder V2 16B (Alternative):**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/deepseek-coder-v2-16b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1
```

**For GLM-4-32B-0414 AWQ (Alternative):**
```bash
# 32K context (native, safe)
python -m vllm.entrypoints.openai.api_server \
    --model ./models/glm-4-32b-0414-awq \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1

# 64K context (with YaRN - requires rope_scaling config in model)
python -m vllm.entrypoints.openai.api_server \
    --model ./models/glm-4-32b-0414-awq \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1
```

**Note**: For 64K context, you may need to add YaRN configuration to the model's `config.json`:
```json
"rope_scaling": {
  "type": "yarn",
  "factor": 2.0,
  "original_max_position_embeddings": 32768
}
```

### 4. Configure Continue Extension

**In VS Code:**
1. Install "Continue" extension
2. Add new model:
   - Provider: **OpenAI**
   - Base URL: `http://localhost:8000/v1`
   - API Key: (leave empty)
   - Model: `Qwen/Qwen2.5-Coder-32B-Instruct` (or your model name)

**Or edit config directly** (`~/.continue/config.json`):
```json
{
  "models": [
    {
      "title": "Qwen2.5-Coder-32B (Local)",
      "provider": "openai",
      "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": ""
    }
  ]
}
```

### 5. Verify Setup

**Test vLLM Server:**
```bash
# Check if server is running
curl http://localhost:8000/v1/models

# Should return JSON with model information
```

**Test in VS Code:**
1. Open a code file in VS Code
2. Use Ctrl+L (or Cmd+L) to open Continue chat
3. Ask a coding question (e.g., "Write a Python function to sort a list")
4. The model should respond using your local vLLM server

**Check GPU Usage:**
```bash
nvidia-smi  # Should show vLLM using GPU memory
```

---

## ⚙️ Configuration Summary

| Component | Choice | Details |
|----------|--------|---------|
| **Runtime** | vLLM | OpenAI-compatible API |
| **IDE** | VS Code | Better local LLM support |
| **Extension** | Continue | Best integration |
| **Primary Model** | Qwen2.5-Coder-32B (AWQ) | Best VRAM utilization |
| **Context Size** | 64K | Sufficient for most tasks |
| **VRAM Usage** | ~26-30GB | 81-94% utilization |
| **Port** | 8000 | vLLM server |

---

## 🔧 Runtime Comparison: vLLM vs TensorRT-LLM

### vLLM (Currently Recommended)

**Overview:**
- Open-source, high-throughput inference engine for LLMs
- Developed by the vLLM team, widely adopted in the community
- Excellent balance of performance, ease of use, and flexibility

**Key Features:**
- ✅ **PagedAttention**: Efficient memory management, reduces fragmentation
- ✅ **Continuous Batching**: Dynamic request processing, maximizes GPU utilization
- ✅ **Easy Setup**: Works directly with Hugging Face models, no compilation needed
- ✅ **OpenAI-Compatible API**: Seamless integration with Continue extension
- ✅ **Active Development**: Frequent updates, broad model support

**Performance:**
- Very good inference speed (30-100 tokens/s depending on model)
- Efficient memory usage with PagedAttention
- Good for development, testing, and production

**Best For:**
- ✅ Development and experimentation
- ✅ Easy model switching
- ✅ Quick setup and deployment
- ✅ Coding tasks and general use

---

### TensorRT-LLM (Alternative)

**Overview:**
- NVIDIA's high-performance inference library for LLMs
- Leverages TensorRT optimizations for maximum throughput on NVIDIA GPUs
- Designed for production workloads with strict performance requirements

**Key Features:**
- ✅ **Hardware-Aware Optimizations**: Fused attention kernels, optimized kernel fusion
- ✅ **Quantization Workflows**: FP8 and INT8 modes for efficient computation
- ✅ **Peak Performance**: 10-20% faster than vLLM in some scenarios
- ✅ **Production-Ready**: Predictable latency and performance profiles
- ⚠️ **Model Compilation**: Requires compiling models before use
- ⚠️ **NVIDIA-Only**: Optimized specifically for NVIDIA hardware

**Performance:**
- Highest inference speed on NVIDIA GPUs
- Lower latency for production workloads
- Better throughput for large batch sizes

**Best For:**
- ✅ Production deployments with strict latency requirements
- ✅ Maximum performance is critical
- ✅ You're willing to compile models
- ✅ You need absolute best performance on NVIDIA GPUs

---

### Comparison Table

| Feature | vLLM | TensorRT-LLM |
|---------|------|--------------|
| **Setup Complexity** | ⭐ Easy | ⭐⭐⭐ Complex (compilation) |
| **Model Compatibility** | ✅ Direct Hugging Face | ⚠️ Requires compilation |
| **Performance** | ⭐⭐⭐⭐ Very Good | ⭐⭐⭐⭐⭐ Excellent |
| **Memory Efficiency** | ✅ PagedAttention | ✅ Hardware-optimized |
| **Development Speed** | ✅ Fast iteration | ⚠️ Slower (compile step) |
| **Production Ready** | ✅ Yes | ✅✅ Yes (optimized) |
| **Hardware Support** | ✅ NVIDIA (primary) | ✅ NVIDIA only |
| **Community Support** | ✅ Large | ✅ NVIDIA-backed |

### Recommendation

**For Your Use Case (Coding with VS Code/Cursor):**
- **✅ vLLM is recommended** because:
  - Simpler setup - no compilation needed
  - Easy model switching for testing
  - Good performance for coding tasks
  - Faster development iteration
  - The performance difference (10-20%) is small for coding tasks

**Consider TensorRT-LLM if:**
- You need maximum performance in production
- Latency is critical
- You're willing to compile models
- You're deploying at scale

**Bottom Line:** vLLM offers the best balance of performance and ease of use for coding tasks. TensorRT-LLM is better for production deployments where every millisecond counts.

---

## 📏 Context Size and VRAM Usage

### How Context Size Affects VRAM

**Yes, larger context windows use MORE VRAM.** Here's why:

**VRAM Components:**
1. **Model Weights** (fixed): ~20GB for 32B AWQ model
2. **KV Cache** (variable): Grows linearly with context length
3. **Activations** (variable): ~2-4GB typically

**KV Cache Growth:**
- The KV (Key-Value) cache stores attention information for each token
- **Linear scaling**: 2x context = 2x KV cache memory
- **Example for 32B model (AWQ)**:
  - 32K context: ~2-3GB KV cache
  - 64K context: ~4-6GB KV cache (2x)
  - 128K context: ~8-10GB KV cache (4x)

**Practical Impact:**
- **32K → 64K**: Adds ~2-3GB VRAM
- **32K → 128K**: Adds ~6-7GB VRAM
- This is why larger contexts may exceed your 32GB VRAM limit

**Recommendations:**
- **Qwen2.5-Coder-32B**: Use 64K max (native support, fits comfortably)
- **DeepSeek Coder V2 16B**: Can use 128K (native support, smaller model = more headroom)
- **GLM-4-32B-0414**: Use 32K native or 64K with YaRN (both fit comfortably), avoid 128K (may exceed 32GB)

**YaRN for Context Extension:**
- **Qwen2.5-Coder-32B**: Already has 64K native - YaRN could extend to 128K+ but would require ~8-10GB KV cache (may exceed 32GB VRAM)
- **DeepSeek Coder V2 16B**: Already has 128K native - YaRN could extend to 256K+ but would require ~16-20GB KV cache (would exceed 32GB VRAM)
- **GLM-4-32B-0414**: Native 32K - YaRN can extend to 64K (recommended, fits in 32GB) or 128K (may exceed 32GB)
- **Bottom Line**: YaRN is compatible with all these models (they use RoPE), but extending beyond native contexts significantly increases VRAM usage

---

## 🔄 Model Switching

### Quick Steps

If you want to switch between models:

1. **Stop current vLLM server** (Ctrl+C in terminal running vLLM)
2. **Stop compression proxy** (if running, Ctrl+C in that terminal)
3. **Start new vLLM server** with different model path and context size
4. **Start compression proxy** (if using, with updated MAX_CONTEXT_TOKENS if needed)
5. **Update Continue config** with new model name
6. **Restart VS Code** (or reload Continue extension)

### Detailed Example: Switching from Qwen2.5-Coder-32B to DeepSeek Coder V2 16B

**Step 1: Stop Current Services**
```bash
# In terminal running vLLM: Ctrl+C
# In terminal running proxy (if used): Ctrl+C
```

**Step 2: Start New vLLM Server**
```bash
# For DeepSeek Coder V2 16B (128K context)
python -m vllm.entrypoints.openai.api_server \
    --model ./models/deepseek-coder-v2-16b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1
```

**Step 3: Update Compression Proxy (if using)**
```python
# In compression_proxy.py, update:
MAX_CONTEXT_TOKENS = 131072  # Changed from 65536 to 131072
```

Then restart proxy:
```bash
python compression_proxy.py
```

**Step 4: Update Continue Config**
Edit `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "DeepSeek Coder V2 16B (Local)",
      "provider": "openai",
      "model": "deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct",
      "apiBase": "http://localhost:8000/v1",  // or 8001 if using proxy
      "apiKey": ""
    }
  ]
}
```

**Step 5: Restart VS Code**
- Close and reopen VS Code, or
- Reload Continue extension: Cmd+Shift+P → "Reload Window"

### Helper Scripts (Recommended)

Create shell scripts to make switching easier:

**`start-qwen.sh`:**
```bash
#!/bin/bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-coder-32b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1
```

**`start-deepseek.sh`:**
```bash
#!/bin/bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/deepseek-coder-v2-16b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 131072 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1
```

**Make executable:**
```bash
chmod +x start-qwen.sh start-deepseek.sh
```

**`start-glm.sh`:**
```bash
#!/bin/bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/glm-4-32b-0414-awq \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.9 \
    --quantization awq \
    --tensor-parallel-size 1
```

**Make executable:**
```bash
chmod +x start-qwen.sh start-deepseek.sh start-glm.sh
```

**Usage:**
```bash
./start-qwen.sh      # Start Qwen2.5-Coder-32B
./start-deepseek.sh  # Start DeepSeek Coder V2 16B
./start-glm.sh       # Start GLM-4-32B-0414 (AWQ)
```

### Important Notes

- **Conversation history resets** when switching models (if using compression proxy)
- **Model must be downloaded** before starting server
- **Context size changes** - update compression proxy config if using
- **Continue config** must match the model name vLLM reports
- **Port conflicts** - ensure old server is stopped before starting new one

---

## 📊 Performance Expectations

### Qwen2.5-Coder-32B (AWQ, 64K context):
- **Inference Speed**: 30-70 tokens/s
- **VRAM Usage**: ~26-30GB
- **Context Loading**: 1-2 seconds for 64K context
- **Response Time**: 2-5 seconds for typical coding queries

### DeepSeek Coder V2 16B (AWQ, 128K context):
- **Inference Speed**: 40-100 tokens/s
- **VRAM Usage**: ~20-24GB
- **Context Loading**: 2-3 seconds for 128K context
- **Response Time**: 2-5 seconds for typical coding queries

### GLM-4-32B-0414 (AWQ, 32K context):
- **Inference Speed**: 30-70 tokens/s
- **VRAM Usage**: ~24-27GB
- **Context Loading**: 1-2 seconds for 32K context
- **Response Time**: 2-5 seconds for typical coding queries
- **Strengths**: Strong reasoning, multi-step code analysis

---

## 🎯 Quick Start Commands

```bash
# 1. Install vLLM
pip install vllm

# 2. Download primary model
huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir ./models/qwen2.5-coder-32b

# 3. Start server
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-coder-32b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --quantization awq

# 4. Configure Continue extension in VS Code
# Base URL: http://localhost:8000/v1
# Model: Qwen/Qwen2.5-Coder-32B-Instruct
```

---

## 📝 Notes

- **Primary model choice**: Qwen2.5-Coder-32B for best VRAM utilization and quality
- **Alternative models**: 
  - DeepSeek Coder V2 16B if you need 128K context
  - GLM-4-32B-0414 (AWQ) for strong reasoning and GPT-4o-level performance
- **All models can be downloaded** and switched as needed
- **vLLM provides OpenAI-compatible API** - works seamlessly with Continue
- **32GB VRAM is well-utilized** with Qwen2.5-Coder-32B (81-94%)

---

## 🔗 Quick Links

- **Qwen2.5-Coder-32B**: https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct
- **DeepSeek Coder V2**: https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct
- **GLM-4-32B-0414 (AWQ)**: https://huggingface.co/AMead10/GLM-4-32B-0414-awq
- **vLLM**: https://github.com/vllm-project/vllm
- **Continue Extension**: https://marketplace.visualstudio.com/items?itemName=Continue.continue
- **LLMLingua**: https://github.com/microsoft/LLMLingua
- **BGE Embeddings**: https://huggingface.co/BAAI/bge-large-en-v1.5

---

## 🗜️ Automatic Context Compression

### Recommended Approach: **Hybrid Semantic Retrieval + LLMLingua**

**Why This Approach:**
- ✅ **Code-aware**: Semantic search preserves code structure and relationships
- ✅ **Efficient**: Only compresses relevant code chunks, not everything
- ✅ **Automatic**: Works transparently with Continue extension
- ✅ **Flexible**: Can handle large codebases and long planning sessions

**How It Works:**
1. **Stage 1 - Semantic Retrieval**: Uses BGE embeddings to find relevant code chunks based on your query
2. **Stage 2 - Compression**: Uses LLMLingua to compress selected chunks if still too large

---

### Deployment: Compression Proxy Middleware

**Architecture:**
```
Continue Extension → Compression Proxy (Port 8001) → vLLM Server (Port 8000)
```

#### Step 1: Install Dependencies

**If using conda environment:**
```bash
conda activate vllm-test
pip install llmlingua sentence-transformers flask requests scikit-learn
```

**Or if using the environment.yaml file, these are already included:**
```bash
conda env create -f environment.yaml  # Creates environment with all dependencies
conda activate vllm-test
```

#### Step 2: Create Compression Proxy

Create `compression_proxy.py`:

```python
from flask import Flask, request, jsonify
import requests
from llmlingua import PromptCompressor
from sentence_transformers import SentenceTransformer
import os
import json

app = Flask(__name__)

# Initialize models
compressor = PromptCompressor()
embedding_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
vllm_url = "http://localhost:8000/v1"

# Configuration
MAX_CONTEXT_TOKENS = 65536  # Your model's max context (64K for Qwen2.5-Coder-32B)
COMPRESSION_THRESHOLD = 0.8  # Auto-compress when 80% full (like Claude Code)
KEEP_FIRST_MESSAGES = 1  # Keep first N messages uncompressed (your dev prompt)
KEEP_RECENT_MESSAGES = 5  # Always keep last N messages uncompressed
SEMANTIC_CHUNKS = 20  # Number of code chunks to retrieve

# Global conversation state (in-memory, resets on restart)
conversation_history = []

@app.route('/v1/chat/completions', methods=['POST'])
def compress_and_forward():
    try:
        data = request.json
        messages = data.get('messages', [])
        
        # Update conversation history
        conversation_history.extend(messages)
        
        # Estimate total token usage
        total_tokens = estimate_tokens(conversation_history)
        
        # Automatic compression when approaching limit (like Claude Code)
        if total_tokens > (MAX_CONTEXT_TOKENS * COMPRESSION_THRESHOLD):
            print(f"⚠️ Context at {total_tokens}/{MAX_CONTEXT_TOKENS} tokens - Auto-compressing...")
            conversation_history = auto_compress_conversation(conversation_history)
        
        # Extract user query and code context from current messages
        user_message = messages[-1]['content'] if messages else ""
        code_context = extract_code_context(messages)
        
        # Stage 1: Semantic retrieval (find relevant code chunks)
        relevant_chunks = semantic_search(user_message, code_context)
        
        # Stage 2: Compress code context if needed
        compressed_code = compress_context(relevant_chunks, MAX_CONTEXT_TOKENS // 2)
        
        # Combine compressed conversation + compressed code
        final_messages = combine_context(conversation_history, compressed_code, user_message)
        
        # Final check: ensure we're under limit
        final_tokens = estimate_tokens(final_messages)
        if final_tokens > MAX_CONTEXT_TOKENS:
            # Emergency compression
            final_messages = emergency_compress(final_messages, MAX_CONTEXT_TOKENS)
        
        data['messages'] = final_messages
        
        # Forward to vLLM
        response = requests.post(f"{vllm_url}/chat/completions", json=data, timeout=120)
        return jsonify(response.json())
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def estimate_tokens(messages):
    """Estimate token count (rough: 1 token ≈ 4 characters)"""
    total_chars = sum(len(str(msg.get('content', ''))) for msg in messages)
    return total_chars // 4

def auto_compress_conversation(messages):
    """Automatically compress conversation history (Claude Code style)"""
    total_to_keep = KEEP_FIRST_MESSAGES + KEEP_RECENT_MESSAGES
    
    if len(messages) <= total_to_keep:
        return messages
    
    # Keep first messages (your dev prompt) + recent messages
    first_messages = messages[:KEEP_FIRST_MESSAGES]
    recent_messages = messages[-KEEP_RECENT_MESSAGES:]
    middle_messages = messages[KEEP_FIRST_MESSAGES:-KEEP_RECENT_MESSAGES]
    
    # Only compress the middle messages (between first and recent)
    if middle_messages:
        middle_content = '\n\n'.join([msg.get('content', '') for msg in middle_messages])
        summarized = summarize_conversation(middle_content)
        
        # Create summary message
        summary_message = {
            "role": "system",
            "content": f"[Previous conversation summary]: {summarized}"
        }
        
        # Return: first (uncompressed) + summary + recent (uncompressed)
        return first_messages + [summary_message] + recent_messages
    else:
        # No middle messages to compress
        return first_messages + recent_messages

def summarize_conversation(text):
    """Summarize conversation using LLMLingua or simple extraction"""
    if len(text) < 1000:
        return text  # Too short to summarize
    
    try:
        # Use LLMLingua to compress/summarize
        compressed = compressor.compress_prompt(
            text,
            target_length=2000,  # Target ~2000 tokens for summary
            rate=0.3  # Aggressive compression for old messages
        )
        return compressed
    except Exception as e:
        print(f"Summarization error: {e}")
        # Fallback: extract first/last parts
        return text[:1000] + "\n\n[... conversation continues ...]\n\n" + text[-1000:]

def combine_context(conversation, code_context, user_query):
    """Combine conversation history with code context"""
    messages = []
    
    # Add conversation history (already compressed if needed)
    messages.extend(conversation)
    
    # Add code context if present
    if code_context:
        messages.append({
            "role": "system",
            "content": f"[Relevant code context]:\n{code_context}"
        })
    
    # Add current user query
    messages.append({
        "role": "user",
        "content": user_query
    })
    
    return messages

def emergency_compress(messages, max_tokens):
    """Emergency compression when still over limit"""
    # Keep only the most recent messages
    current_tokens = estimate_tokens(messages)
    if current_tokens <= max_tokens:
        return messages
    
    # Always keep first messages (dev prompt) uncompressed
    first_messages = messages[:KEEP_FIRST_MESSAGES]
    remaining_messages = messages[KEEP_FIRST_MESSAGES:]
    
    # Calculate tokens for first messages
    first_tokens = estimate_tokens(first_messages)
    remaining_budget = max_tokens - first_tokens
    
    if remaining_budget <= 0:
        # Even first messages are too large - keep only first (shouldn't happen)
        return first_messages[:1]  # Keep at least first message
    
    # Calculate how many remaining messages to keep
    remaining_tokens = estimate_tokens(remaining_messages)
    if remaining_tokens <= remaining_budget:
        return first_messages + remaining_messages
    
    ratio = remaining_budget / remaining_tokens
    keep_count = max(KEEP_RECENT_MESSAGES, int(len(remaining_messages) * ratio))
    
    # Keep system messages + recent messages from remaining
    system_msgs = [m for m in remaining_messages if m.get('role') == 'system']
    other_msgs = [m for m in remaining_messages if m.get('role') != 'system']
    recent_other = other_msgs[-keep_count:] if len(other_msgs) > keep_count else other_msgs
    
    return first_messages + system_msgs + recent_other

def extract_code_context(messages):
    """Extract code context from messages"""
    context = []
    for msg in messages:
        if msg.get('role') in ['system', 'user', 'assistant']:
            content = msg.get('content', '')
            # Extract code blocks
            if '```' in content:
                context.append(content)
    return '\n\n'.join(context)

def semantic_search(query, codebase):
    """Use BGE embeddings to find relevant code chunks"""
    if not codebase:
        return []
    
    # Simple chunking (split by functions/classes)
    chunks = chunk_code(codebase)
    
    if len(chunks) <= SEMANTIC_CHUNKS:
        return chunks
    
    try:
        # Encode query and chunks
        query_embedding = embedding_model.encode(query)
        chunk_embeddings = embedding_model.encode(chunks)
        
        # Calculate similarities
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get top chunks
        top_indices = similarities.argsort()[-SEMANTIC_CHUNKS:][::-1]
        return [chunks[i] for i in top_indices]
    except Exception as e:
        print(f"Semantic search error: {e}, returning all chunks")
        return chunks[:SEMANTIC_CHUNKS]

def chunk_code(code):
    """Split code into semantic chunks (functions, classes, etc.)"""
    # Simple implementation: split by function/class definitions
    import re
    chunks = []
    current_chunk = []
    
    lines = code.split('\n')
    for line in lines:
        current_chunk.append(line)
        # Check for function/class boundaries
        if re.match(r'^(def |class |async def )', line.strip()):
            if current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = [line]
    
    if current_chunk:
        chunks.append('\n'.join(current_chunk))
    
    return chunks if chunks else [code]

def compress_context(chunks, max_tokens):
    """Compress context using LLMLingua if needed"""
    if not chunks:
        return ""
    
    combined = '\n\n'.join(chunks)
    
    # Estimate tokens (rough: 1 token ≈ 4 characters)
    estimated_tokens = len(combined) // 4
    
    if estimated_tokens > max_tokens:
        # Compress to target length
        try:
            compressed = compressor.compress_prompt(
                combined, 
                target_length=max_tokens,
                rate=0.5  # Compression rate
            )
            return compressed
        except Exception as e:
            print(f"Compression error: {e}, returning truncated")
            # Fallback: truncate
            return combined[:max_tokens * 4]
    return combined

# Note: reconstruct_messages function removed - now using combine_context instead

if __name__ == '__main__':
    print("Starting compression proxy on http://127.0.0.1:8001")
    print(f"Forwarding to vLLM at {vllm_url}")
    app.run(host='127.0.0.1', port=8001, debug=False)
```

#### Step 3: Start Services

**Terminal 1 - Start vLLM:**
```bash
python -m vllm.entrypoints.openai.api_server \
    --model ./models/qwen2.5-coder-32b \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 65536 \
    --quantization awq
```

**Terminal 2 - Start Compression Proxy:**
```bash
python compression_proxy.py
```

#### Step 4: Update Continue Configuration

Update `~/.continue/config.json` to use the proxy:

```json
{
  "models": [
    {
      "title": "Qwen2.5-Coder-32B (Compressed)",
      "provider": "openai",
      "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
      "apiBase": "http://localhost:8001/v1",
      "apiKey": ""
    }
  ]
}
```

**Note**: Change port from `8000` to `8001` to use the compression proxy.

---

### Configuration Options

You can adjust these in `compression_proxy.py`:

- **MAX_CONTEXT_TOKENS**: Target context size (default: 65536)
- **COMPRESSION_THRESHOLD**: When to compress (default: 0.8 = 80%)
- **KEEP_FIRST_MESSAGES**: Keep first N messages uncompressed (default: 1)
- **KEEP_RECENT_MESSAGES**: Keep last N messages uncompressed (default: 5)
- **SEMANTIC_CHUNKS**: Number of code chunks to retrieve (default: 20)
- **Compression rate**: How aggressive to compress (default: 0.5)

---

### Automatic Context Management (Claude Code Style)

**How It Works:**
1. **Monitors context size** on every request
2. **Auto-compresses when 80% full** (configurable threshold)
3. **Keeps recent messages** (last 5 messages always uncompressed)
4. **Summarizes old conversation** using LLMLingua
5. **Preserves code context** through semantic search

**Behavior:**
- ✅ **Automatic**: No manual intervention needed
- ✅ **Proactive**: Compresses before hitting limits
- ✅ **Smart**: Keeps recent context, summarizes old
- ✅ **Transparent**: Works seamlessly with Continue

**Configuration:**
- `COMPRESSION_THRESHOLD = 0.8` - Compress at 80% capacity
- `KEEP_FIRST_MESSAGES = 1` - Always keep first message(s) uncompressed (your dev prompt)
- `KEEP_RECENT_MESSAGES = 5` - Always keep last 5 messages uncompressed
- `MAX_CONTEXT_TOKENS = 65536` - Your model's max context

**What is a "Message"?**
- A **message** is a single entry in the conversation array (not a request)
- Each message has a `role` (system/user/assistant) and `content`
- **Length doesn't matter** - a 2k line prompt in one message = 1 message
- **One request can contain multiple messages**

**Examples:**
```json
// This is 1 message (even if content is 2k lines):
{
  "role": "user",
  "content": "Your entire 2000 line dev prompt here..."
}

// This is 2 messages:
[
  {"role": "system", "content": "You are a coding assistant"},
  {"role": "user", "content": "Your dev prompt..."}
]
```

**Special Feature - Preserve Initial Prompt:**
- Your starting development prompt (~2k lines) is **always kept uncompressed**
- **If your prompt is in 1 message**: `KEEP_FIRST_MESSAGES = 1` ✅
- **If your prompt spans 2 messages**: `KEEP_FIRST_MESSAGES = 2` (e.g., system + user)
- **If your prompt spans 3 messages**: `KEEP_FIRST_MESSAGES = 3`, etc.
- Only messages between first and recent get compressed
- This ensures your dev prompt stays intact throughout the session

### Benefits

- ✅ **Automatic**: Works transparently with Continue extension
- ✅ **Code-aware**: Semantic search preserves code structure
- ✅ **Efficient**: Only compresses when needed
- ✅ **Flexible**: Can handle large codebases and long planning sessions
- ✅ **No vLLM changes**: Works with existing setup
- ✅ **Claude-like**: Automatic context management during long sessions

---

### Alternative: Simple LLMLingua (No Semantic Search)

If you want simpler setup without semantic search:

```bash
pip install llmlingua
```

Then modify the proxy to just compress all context without semantic filtering. This is simpler but less code-aware.

---

## 🔧 Troubleshooting

### vLLM Issues

**Out of Memory Error:**
```bash
# Reduce context size
--max-model-len 32768  # Instead of 65536

# Reduce GPU memory utilization
--gpu-memory-utilization 0.8  # Instead of 0.9
```

**Model Not Loading:**
- Check CUDA: `nvidia-smi`
- Verify model path is correct
- Check VRAM: `nvidia-smi` (should show available memory)
- Ensure model files are complete (check file sizes)

**Slow Performance:**
- Use AWQ quantization (faster than FP16)
- Reduce context window size
- Close other GPU applications
- Check GPU utilization: `nvidia-smi`

**AWQ Model Not Found:**
- Some models may not have pre-quantized AWQ versions
- You may need to use FP16 or quantize yourself
- Check Hugging Face model page for available formats

### Continue Extension Issues

**Not Connecting:**
- Verify vLLM server is running: `curl http://localhost:8000/v1/models`
- Check base URL in Continue settings: `http://localhost:8000/v1`
- Restart VS Code
- Check Continue extension logs

**Wrong Model Name:**
- The model name in Continue config should match what vLLM reports
- Check: `curl http://localhost:8000/v1/models` to see actual model name
- Update Continue config with correct model name

### Compression Proxy Issues

**Proxy Not Starting:**
- Check if port 8001 is available: `lsof -i :8001`
- Verify all dependencies installed: `pip list | grep llmlingua`
- Check Python version: `python --version` (needs 3.10+)

**Compression Errors:**
- LLMLingua may need to download models on first use (be patient)
- Check proxy logs for error messages
- Try simpler compression (disable semantic search)

**Conversation History Lost:**
- Note: Conversation history is in-memory and resets when proxy restarts
- This is by design (stateless proxy)
- For persistence, you'd need to add database/file storage

---

## ⚠️ Important Notes

### Model Format Availability

- **AWQ quantization**: Not all models have pre-quantized AWQ versions
- If AWQ not available, use FP16 (will use more VRAM)
- Check Hugging Face model page for available formats
- You may need to quantize models yourself if AWQ not available

### Conversation History

- **In-memory only**: Conversation history resets when compression proxy restarts
- This is normal behavior for the proxy
- For persistence across restarts, you'd need to add file/database storage
- vLLM itself doesn't maintain conversation state

### Model Download

- Models are large (20-40GB), download may take time
- Ensure stable internet connection
- Use `huggingface-cli` for resumable downloads
- Check disk space before downloading

---

## ✅ Checklist

- [ ] vLLM installed
- [ ] Model downloaded (Qwen2.5-Coder-32B or DeepSeek Coder V2 16B)
- [ ] vLLM server running on port 8000
- [ ] Continue extension installed in VS Code
- [ ] Continue configured with local vLLM endpoint
- [ ] Tested with a coding query
- [ ] (Optional) Context compression proxy installed and running
- [ ] (Optional) Continue configured to use compression proxy (port 8001)
- [ ] Verified GPU usage with `nvidia-smi`
- [ ] Tested with coding queries
- [ ] (Optional) Tested compression proxy with long conversation

---

**This document represents our final, definitive choices. Update as decisions are made.**

