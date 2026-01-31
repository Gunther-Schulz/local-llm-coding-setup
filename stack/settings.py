"""
Central configuration for the entire system.

All configurable settings for vLLM, proxy, and vision are defined here.
Can be overridden via environment variables.

Configuration priority (highest to lowest):
1. Environment variables (runtime)
2. config/settings.env file (user edits)
3. Defaults in this file
"""
import os
from pathlib import Path


def _load_env_file():
    """Load settings from config/settings.env if it exists."""
    # Find project root
    current = Path(__file__).resolve()
    root_path = current.parents[1]  # Go up from stack/ to project root
    
    env_file = root_path / "config" / "settings.env"
    if not env_file.exists():
        return
    
    # Parse and load into os.environ (only if not already set)
    with open(env_file) as f:
        for line in f:
            line = line.strip()
            # Skip comments and empty lines
            if not line or line.startswith("#"):
                continue
            
            # Parse KEY=VALUE
            if "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip()
                
                # Only set if not already in environment
                if key and key not in os.environ:
                    os.environ[key] = value


# Load config file first (before defining settings)
_load_env_file()


# ============================================================================
# Server Ports and URLs
# ============================================================================

VLLM_HOST = os.getenv("VLLM_HOST", "0.0.0.0")
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_URL = os.getenv("VLLM_URL", f"http://localhost:{VLLM_PORT}")

# LLM backend URL (vLLM or llama-server); proxy and clients use this
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{VLLM_PORT}")

PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8002"))

VISION_HOST = os.getenv("VISION_HOST", "0.0.0.0")
VISION_PORT = int(os.getenv("VISION_PORT", "8004"))
VISION_URL = os.getenv("VISION_API_URL", f"http://localhost:{VISION_PORT}")


# ============================================================================
# vLLM Configuration
# ============================================================================

VLLM_DTYPE = os.getenv("VLLM_DTYPE", "float16")
VLLM_CPU_OFFLOAD_GB = os.getenv("VLLM_CPU_OFFLOAD_GB", "8")
VLLM_TENSOR_PARALLEL = int(os.getenv("VLLM_TENSOR_PARALLEL", "1"))

# CUDA graph mode: "PIECEWISE" (safe on 5090+AMD) or "FULL" (may crash)
VLLM_CUDAGRAPH_MODE = os.getenv("VLLM_CUDAGRAPH_MODE", "PIECEWISE")


# ============================================================================
# Context and Token Management
# ============================================================================

# Model context limits (set by model config, can override)
MODEL_MAX_CONTEXT = int(os.getenv("MODEL_MAX_CONTEXT", "32768"))
MODEL_EXTENDED_CONTEXT = os.getenv("MODEL_EXTENDED_CONTEXT")  # Optional extended context

# Prompt limits
MAX_PROMPT_TOKENS = int(os.getenv("MAX_PROMPT_TOKENS", "30000"))  # Max prompt size
SAFETY_MARGIN = int(os.getenv("SAFETY_MARGIN", "1536"))  # Reserve for response


# ============================================================================
# Compression Settings
# ============================================================================

# Context management (Cursor-style: summarization + sliding window)
COMPRESSION_THRESHOLD = int(os.getenv("COMPRESSION_THRESHOLD", "15000"))
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "20"))
MAX_ARCHIVE_MESSAGES = int(os.getenv("MAX_ARCHIVE_MESSAGES", "200"))


# ============================================================================
# Tool Calling
# ============================================================================

# Tool parser format: "openai", "qwen2.5", "qwen3", "auto"
MODEL_TOOL_FORMAT = os.getenv("MODEL_TOOL_FORMAT", "openai")


# ============================================================================
# Vision Configuration
# ============================================================================

# Vision model paths (set by vision launcher from model config)
VISION_GGUF_PATH = os.getenv("VISION_GGUF_PATH", "")
VISION_MMPROJ_PATH = os.getenv("VISION_MMPROJ_PATH", "")
VISION_MAX_CONTEXT = int(os.getenv("VISION_MAX_CONTEXT", "32768"))

# llama.cpp binary path (use mtmd-cli - all specialized binaries are deprecated)
LLAMACPP_BIN = os.getenv("LLAMACPP_BIN", "./external/llama.cpp/build/bin/llama-mtmd-cli")

# llama-server for LLM engine (when LLM_ENGINE=llamacpp); CUDA build in external/llama.cpp/build-cuda
LLAMACPP_SERVER_BIN = os.getenv("LLAMACPP_SERVER_BIN", "./external/llama.cpp/build-cuda/bin/llama-server")


# ============================================================================
# Debug and Logging
# ============================================================================

DEBUG = os.getenv("DEBUG", "0") == "1"
LOG_DIR = os.getenv("LOG_DIR", "logs")


# ============================================================================
# Helper Functions
# ============================================================================

def get_effective_context_limit() -> int:
    """Get the effective context limit for the current model."""
    # Check if extended context is enabled
    extended_ctx = MODEL_EXTENDED_CONTEXT
    if extended_ctx:
        return int(extended_ctx)
    return MODEL_MAX_CONTEXT


def get_max_completion_tokens(prompt_tokens: int) -> int:
    """Calculate maximum completion tokens given prompt size."""
    ctx_limit = get_effective_context_limit()
    return max(1, ctx_limit - SAFETY_MARGIN - prompt_tokens)


def print_config_summary():
    """Print current configuration (useful for debugging)."""
    print("=" * 70)
    print("Configuration Summary")
    print("=" * 70)
    print(f"Backend:     {BACKEND_URL}")
    print(f"vLLM:        {VLLM_URL}")
    print(f"Proxy:       {PROXY_HOST}:{PROXY_PORT}")
    print(f"Vision:      {VISION_URL}")
    print(f"Context:     {MODEL_MAX_CONTEXT} tokens")
    print(f"Max Prompt:  {MAX_PROMPT_TOKENS} tokens")
    print(f"Context Mgmt: Cursor-style (summarization + window)")
    print(f"  Threshold:    {COMPRESSION_THRESHOLD} tokens (triggers at 30 msgs OR this)")
    print(f"  Window Size:  {CONTEXT_WINDOW_SIZE} messages (keep recent)")
    print(f"  Max Archive:  {MAX_ARCHIVE_MESSAGES} messages")
    print(f"Debug:       {'On' if DEBUG else 'Off'}")
    print("=" * 70)
