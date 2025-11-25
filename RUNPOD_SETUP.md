# RunPod Setup Guide for vLLM Coding Models

> **Purpose**: Test RTX 5090 performance on RunPod to decide whether to buy one locally

---

## 🚀 Quick Start: RunPod Dockerless CLI

### Prerequisites

1. **RunPod Account**: Sign up at https://www.runpod.io
2. **API Key**: Get your API key from RunPod dashboard
3. **runpodctl CLI**: Install the RunPod CLI tool

### Step 1: Install runpodctl

```bash
# Install runpodctl (version 1.11.0+ required for Dockerless workflow)
pip install runpod

# Or using the official installer
curl -sSL https://runpod.io/install | bash
```

### Step 2: Configure runpodctl

```bash
# Configure with your API key
runpodctl config

# Or set it directly
export RUNPOD_API_KEY="your-api-key-here"
```

### Step 3: Create a RunPod Project

```bash
# Create a new project for vLLM
runpodctl project create vllm-coding-setup

# This will:
# - Create project directory structure
# - Generate runpod.toml config file
# - Create src/handler.py for your code
# - Create builder/requirements.txt for dependencies
```

### Step 4: Project Structure

After creating the project, you'll have:

```
vllm-coding-setup/
├── .runpodignore          # Files to exclude from deployment
├── runpod.toml           # Project configuration
├── builder/
│   └── requirements.txt   # Python dependencies
└── src/
    └── handler.py        # Your serverless handler code
```

---

## 📝 Configure Your Project

### Update `runpod.toml`

```toml
[project]
name = "vllm-coding-setup"
base_image = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel"

[build]
requirements_path = "builder/requirements.txt"

[deploy]
handler = "src/handler.py"
```

### Update `builder/requirements.txt`

```txt
vllm>=0.11.2
huggingface-hub>=0.36.0
```

### Create `src/handler.py`

```python
import runpod
from vllm import LLM, SamplingParams

# Initialize model (will be loaded once on cold start)
model = None

def handler(event):
    """
    Handler for RunPod serverless endpoint
    """
    global model
    
    # Load model on first request (cold start)
    if model is None:
        model_id = event.get("model", "Qwen/Qwen2.5-Coder-32B-Instruct")
        model = LLM(
            model=model_id,
            max_model_len=65536,
            gpu_memory_utilization=0.9,
            quantization="awq"  # If using AWQ model
        )
    
    # Get request data
    prompt = event.get("prompt", "")
    max_tokens = event.get("max_tokens", 512)
    temperature = event.get("temperature", 0.7)
    
    # Generate response
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens
    )
    
    outputs = model.generate([prompt], sampling_params)
    
    return {
        "response": outputs[0].outputs[0].text,
        "model": model_id
    }

# Start the serverless worker
runpod.serverless.start({"handler": handler})
```

---

## 🧪 Development Workflow

### Start Development Session

```bash
# Start a development pod (interactive session)
runpodctl project dev

# This will:
# - Create a pod with your project
# - Sync your local code to the pod
# - Provide a URL for testing
# - Show logs in real-time
```

### Test Your Handler

Once the dev session is running, you'll get a URL like:
```
https://your-endpoint-id.runpod.net
```

Test it with:
```bash
curl -X POST https://your-endpoint-id.runpod.net \
  -H "Content-Type: application/json" \
  -d '{
    "input": {
      "prompt": "Write a Python function to sort a list",
      "max_tokens": 200
    }
  }'
```

---

## 🚀 Deploy to Production

### Deploy as Serverless Endpoint

```bash
# Deploy your project as a serverless endpoint
runpodctl project deploy

# This will:
# - Upload your code and dependencies
# - Create a serverless endpoint
# - Return endpoint URL
```

### Deploy to Pod (For RTX 5090 Testing)

If you want to test on a specific GPU (like RTX 5090):

```bash
# Create a pod with RTX 5090
runpodctl pod create \
  --name vllm-5090-test \
  --image runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel \
  --gpu-type RTX_5090 \
  --env "MODEL=Qwen/Qwen2.5-Coder-32B-Instruct"

# SSH into the pod
runpodctl pod ssh <pod-id>

# Then manually set up (or use the project files)
```

---

## 🎯 Testing RTX 5090 Performance

### Recommended Setup for Testing

1. **Create Pod with RTX 5090**:
   ```bash
   # Use RunPod console or CLI to create pod
   # Select: RTX 5090 (32GB VRAM)
   ```

2. **Clone Your Repository**:
   ```bash
   git clone https://github.com/Gunther-Schulz/local-llm-coding-setup.git
   cd local-llm-coding-setup
   ```

3. **Set Up Environment**:
   ```bash
   # Create conda environment
   conda env create -f environment.yaml
   conda activate vllm-test
   ```

4. **Download Production Model**:
   ```bash
   # Download the model you want to test
   huggingface-cli download Qwen/Qwen2.5-Coder-32B-Instruct --local-dir ./models/qwen2.5-coder-32b
   # Or GLM-4-32B-0414 AWQ
   huggingface-cli download AMead10/GLM-4-32B-0414-awq --local-dir ./models/glm-4-32b-0414-awq
   ```

5. **Start vLLM Server**:
   ```bash
   # For Qwen2.5-Coder-32B
   python -m vllm.entrypoints.openai.api_server \
       --model ./models/qwen2.5-coder-32b \
       --host 0.0.0.0 \
       --port 8000 \
       --max-model-len 65536 \
       --gpu-memory-utilization 0.9 \
       --quantization awq \
       --tensor-parallel-size 1
   ```

6. **Test Performance**:
   ```bash
   # Benchmark inference speed
   python benchmark.py
   
   # Test with coding queries
   curl http://localhost:8000/v1/chat/completions \
       -H "Content-Type: application/json" \
       -d '{
           "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
           "messages": [{"role": "user", "content": "Write a Python function to calculate fibonacci"}]
       }'
   ```

---

## 📊 Performance Comparison

### What to Measure

1. **Inference Speed**: Tokens per second
2. **VRAM Usage**: Actual memory consumption
3. **Context Loading Time**: Time to load 64K context
4. **Response Latency**: Time to first token
5. **Throughput**: Requests per second

### Benchmark Script

Create `benchmark.py`:

```python
import time
import requests
import json

def benchmark(endpoint_url, num_requests=10):
    """Benchmark the vLLM endpoint"""
    
    prompt = "Write a Python function to sort a list and explain it."
    
    times = []
    tokens_per_second = []
    
    for i in range(num_requests):
        start = time.time()
        
        response = requests.post(
            f"{endpoint_url}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen2.5-Coder-32B-Instruct",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500
            },
            timeout=60
        )
        
        elapsed = time.time() - start
        times.append(elapsed)
        
        if response.status_code == 200:
            data = response.json()
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            if tokens > 0:
                tps = tokens / elapsed
                tokens_per_second.append(tps)
        
        print(f"Request {i+1}/{num_requests}: {elapsed:.2f}s")
    
    print(f"\nAverage latency: {sum(times)/len(times):.2f}s")
    if tokens_per_second:
        print(f"Average tokens/s: {sum(tokens_per_second)/len(tokens_per_second):.2f}")
    
    return {
        "avg_latency": sum(times)/len(times),
        "avg_tokens_per_second": sum(tokens_per_second)/len(tokens_per_second) if tokens_per_second else 0
    }

if __name__ == "__main__":
    endpoint = "http://localhost:8000"  # Or your RunPod endpoint
    benchmark(endpoint)
```

---

## 🔧 Alternative: Use RunPod's Pre-built vLLM Template

RunPod also offers a pre-configured vLLM serverless worker:

1. **Go to RunPod Console** → Serverless → Quick Deploy
2. **Select "Serverless vLLM"**
3. **Configure**:
   - Model: `Qwen/Qwen2.5-Coder-32B-Instruct` or `AMead10/GLM-4-32B-0414-awq`
   - Max Model Length: `65536` (for Qwen) or `32768` (for GLM-4)
   - GPU: RTX 5090 (or A100 for comparison)
4. **Deploy** and test

---

## 💡 Tips for Testing

1. **Compare GPUs**: Test on RTX 5090, A100, and H100 to compare
2. **Test Different Models**: Try Qwen2.5-Coder-32B, GLM-4-32B-0414, DeepSeek Coder V2
3. **Measure Real Usage**: Test with actual coding tasks, not just benchmarks
4. **Cost Analysis**: Calculate cost per hour vs. buying RTX 5090
5. **Network Latency**: Consider latency if accessing remotely

---

## 📝 Next Steps

1. Install `runpodctl` and configure API key
2. Create a RunPod project
3. Set up the handler code
4. Start a dev session to test
5. Deploy to RTX 5090 pod for performance testing
6. Compare results with local RTX 4060
7. Make purchase decision based on performance/cost analysis

---

## 🔗 Resources

- **RunPod Docs**: https://docs.runpod.io
- **runpodctl CLI**: https://github.com/runpod/runpodctl
- **Dockerless Workflow**: https://www.runpod.io/blog/dockerless-cli-runpod
- **vLLM on RunPod**: https://docs.runpod.io/serverless/workers/vllm/get-started

