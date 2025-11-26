# NOTES
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# RunPod Setup: Qwen2.5-Coder-14B-Instruct + Compression Proxy

Use conda

## Model Download

**Qwen2.5-Coder-14B-Instruct Q4_K_M (~9 GB):**
- https://huggingface.co/yemiao2745/Qwen2.5-Coder-14B-Instruct-Q4_K_M-GGUF
- Download to: `/workspace/models/qwen2.5-coder-14b-q4_k_m/`

## Installation

**Important:** Use latest versions of all dependencies to avoid compatibility issues (especially Jinja2 >=3.1.0 for GLM-4 chat template support).

### 1. Conda Environment Setup

```bash
# Install miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p /workspace/miniconda3
source /workspace/miniconda3/etc/profile.d/conda.sh

# Accept terms and create environment
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
conda create -n glm4 python=3.10 -y
conda activate glm4
```

### 2. Install Dependencies

```bash
conda activate glm4
# Install latest llama-cpp-python with CUDA support (pre-built wheel)
pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --no-cache-dir

# Install latest server and other dependencies
pip install --upgrade 'jinja2>=3.1.0' 'fastapi>=0.109.0' 'uvicorn[standard]>=0.27.0' 'llmlingua>=0.2.1' 'sentence-transformers>=2.3.0' 'requests>=2.31.0' 'hf_transfer>=0.1.5' 'pydantic>=2.0.0'
```

### 3. Alternative: Install from requirements.txt

```bash
conda activate glm4
# Install llama-cpp-python with CUDA support first
pip install --upgrade llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu121 --no-cache-dir
# Then install other dependencies
pip install --upgrade -r requirements.txt
```

### 4. Download Model

```bash
cd /workspace
mkdir -p models/qwen2.5-coder-14b-q4_k_m
hf download yemiao2745/Qwen2.5-Coder-14B-Instruct-Q4_K_M-GGUF --include "qwen2.5-coder-14b-instruct-q4_k_m.gguf" --local-dir /workspace/models/qwen2.5-coder-14b-q4_k_m
```

## Components

1. **llama-cpp-python server** (port 8000)
   - Model: Qwen2.5-Coder-14B-Instruct Q4_K_M
   - Context: 65536 tokens (64K) - can support up to 128K
   - GPU layers: All layers on GPU (`--n_gpu_layers -1`)
   - Batch sizes: `--n_batch 256 --n_ubatch 256` (safe for 32GB VRAM)
   - Tool calling: Native support with excellent performance

2. **Compression proxy** (port 8002)
   - FastAPI server
   - Compresses old messages when context > 100K
   - Caches compressed messages
   - Keeps last 3 messages uncompressed
   - Uses smaller compression model: `microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank`
   - **Tool calling support:**
     - Explicitly handles `tools` and `tool_choice` parameters
     - Preserves tool calls and tool responses (never compressed)
     - Passes through all tool calling parameters to llama-cpp-python
     - Supports full OpenAI-compatible tool calling API

3. **Cursor configuration** (local machine)
   - Base URL: RunPod HTTP service URL for port 8002 (e.g., `https://xxxxx-8002.proxy.runpod.net/v1`)
   - API Key: `not-needed`
   - Model ID: Query via `curl https://xxxxx-8002.proxy.runpod.net/v1/models` after server starts
     - Extract ID: `curl https://xxxxx-8002.proxy.runpod.net/v1/models | grep -o '"id":"[^"]*"' | head -1`
     - Or use jq: `curl https://xxxxx-8002.proxy.runpod.net/v1/models | jq -r '.data[0].id'`
   - Tool calling: Supported (Qwen2.5-Coder-14B has native function calling)

## File Structure

```
/workspace/
├── models/qwen2.5-coder-14b-q4_k_m/
│   └── qwen2.5-coder-14b-instruct-q4_k_m.gguf
├── compression_proxy.py
├── start-llama-server.sh
├── start-compression-proxy.sh
├── start-all.sh
└── stop-all.sh
```

## Startup

1. Start llama-cpp-python server (port 8000)
2. Start compression proxy (port 8002) - Note: port 8001 is used by nginx
3. Get RunPod HTTP service URL for port 8002 from RunPod dashboard
4. Configure Cursor (local): Base URL = RunPod HTTP service URL + `/v1`

**Start servers:**
```bash
./start-all.sh
```

**Stop servers:**
```bash
./stop-all.sh
```

## VRAM Usage

- Model: ~9 GB
- KV Cache (64K): ~6–8 GB
- Compute buffer & overhead: ~6–8 GB
- Total: ~21–25 GB (fits easily in 32GB VRAM, even at 64K context)

## Qwen2.5-Coder Performance

Based on benchmarks:
- **HumanEval:** 92.7% (matches GPT-4o)
- **MBPP:** 90.2%
- **Supports 40+ programming languages**
- **Excellent at code completion, refactoring, and debugging**

## Tool Calling Optimization Settings

For optimal performance:

**Recommended parameters:**
- `--temperature 0.15` (lower for accurate tool selection)
- `--top-p 0.7` (restrained exploration)
- `--repeat-penalty 1.2` (prevent tool calling loops)
- `--num-keep 1024` (retain longer history for tool context)
- `--min-p 0.03` (slightly higher for creative tool combinations)
- `--max-tokens 16384` or `32768` (allow space for tool responses)

**Note:** These settings prioritize tool calling accuracy over creativity. Adjust based on your needs.
