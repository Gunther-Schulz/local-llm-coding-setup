# NOTES
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# RunPod Setup: Qwen2.5-Coder-14B-Instruct + Native llama.cpp + Compression Proxy

**Current Setup:** Native llama.cpp server (not llama-cpp-python)

**Future Plans:** Test GLM-4-9B-Chat for Chinese support and alternative tool calling

Use conda

## Model Download

**Qwen2.5-Coder-14B-Instruct Q4_K_M (~9 GB):**
- https://huggingface.co/yemiao2745/Qwen2.5-Coder-14B-Instruct-Q4_K_M-GGUF
- Download to: `/workspace/models/qwen2.5-coder-14b-q4_k_m/`

## Installation

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

### 2. Install Dependencies (for compression proxy only)

```bash
conda activate glm4
# Install dependencies for compression proxy
pip install --upgrade 'fastapi>=0.109.0' 'uvicorn[standard]>=0.27.0' 'llmlingua>=0.2.1' 'sentence-transformers>=2.3.0' 'requests>=2.31.0' 'hf_transfer>=0.1.5' 'pydantic>=2.0.0'
```

### 3. Build Native llama.cpp Server

```bash
cd /workspace
# This will clone llama.cpp, detect GPU, and build with CUDA + ccache
./build-native-llama.sh
```

### 4. Download Model

```bash
cd /workspace
mkdir -p models/qwen2.5-coder-14b-q4_k_m
hf download yemiao2745/Qwen2.5-Coder-14B-Instruct-Q4_K_M-GGUF --include "qwen2.5-coder-14b-instruct-q4_k_m.gguf" --local-dir /workspace/models/qwen2.5-coder-14b-q4_k_m
```

## Components

1. **Native llama-server** (port 8000)
   - Model: Qwen2.5-Coder-14B-Instruct Q4_K_M
   - Context: 81920 tokens (80K) with `--ctx-size 81920`
   - **Actual usable context: 32768 tokens** (slots are auto-capped to model's training context)
   - GPU layers: All layers on GPU (`--n-gpu-layers -1`)
   - Batch sizes: `--batch-size 128 --ubatch-size 128`
   - Flash attention: Enabled (`--flash-attn on`)
   - Continuous batching: Enabled (`--cont-batching`)
   - Tool calling: Native Jinja2 support with `--jinja` flag
   - **Built with:** CMake + CUDA + ccache for faster recompilation

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
├── llama.cpp-native/              # Built native llama.cpp
│   └── build/bin/llama-server
├── compression_proxy.py
├── build-native-llama.sh
├── start-llama-server-native.sh
├── start-compression-proxy.sh
├── start-all-native.sh
└── stop-all.sh
```

## Startup

1. Build native llama.cpp (first time only): `./build-native-llama.sh`
2. Start native llama-server (port 8000)
3. Start compression proxy (port 8002) - Note: port 8001 is used by nginx
4. Get RunPod HTTP service URL for port 8002 from RunPod dashboard
5. Configure Cursor/Continue (local): Base URL = RunPod HTTP service URL + `/v1`

**Start servers:**
```bash
./start-all-native.sh
```

**Stop servers:**
```bash
./stop-all.sh
```

## VRAM Usage (80K context allocation, 32K actual slots)

- Model: ~8.5 GB
- KV Cache (80K allocation): ~15.4 GB (but slots capped at 32K)
- Compute buffer & overhead: ~0.5 GB
- Total: ~24 GB (fits in 32GB VRAM with headroom)

## Qwen2.5-Coder Performance

Based on benchmarks:
- **HumanEval:** 92.7% (matches GPT-4o)
- **MBPP:** 90.2%
- **Supports 40+ programming languages**
- **Excellent at code completion, refactoring, and debugging**

## Known Issues

### Prompt Cache Bug (Native llama.cpp with --jinja)

When using native `llama-server` with `--jinja` flag for tool calling, there's a known bug:

**Symptom:** Server crashes with `Invalid diff: now finding less tool calls!` during multi-turn tool calling

**Root Cause:** 
- Prompt cache enabled by default (8GB limit)
- When cache reloads mid-conversation during streaming tool responses
- The streaming tool call diff tracker gets confused and crashes

**Workaround:** Add `--cache-ram 0` to `start-llama-server-native.sh` to disable prompt cache

**Trade-off:**
- ✅ No crashes with tool calling
- ❌ Slightly slower for repeated similar prompts (but compression proxy mitigates this)

**Status:** Bug reported to llama.cpp maintainers

### Context Window

- **Configured:** 81920 tokens (`--ctx-size 81920`)
- **Actual:** Server auto-caps slots to 32768 (model's training context)
- **Reason:** Qwen2.5-Coder-14B is trained on 32K context
- **Solution:** Compression proxy manages conversation history to fit within 32K
