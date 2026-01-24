"""Configuration and environment variables"""
import os

# Debug mode
DEBUG_MODE = os.environ.get("DEBUG", "0") == "1"

# Model-specific configuration
MODEL_TOOL_FORMAT = os.environ.get("MODEL_TOOL_FORMAT", "auto")  # qwen2.5, qwen3, none, auto
MODEL_MAX_CONTEXT = int(os.environ.get("MODEL_MAX_CONTEXT", "32768"))

# Backend server
BACKEND_SERVER_URL = "http://localhost:8000"

# Dynamic context limits
COMPRESSION_THRESHOLD = int(MODEL_MAX_CONTEXT * 0.75)  # Start compressing at 75%
KEEP_RECENT_MESSAGES = 4
MAX_PROMPT_TOKENS = int(MODEL_MAX_CONTEXT * 0.92)  # Leave 8% for completion

# Tiktoken availability
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    if DEBUG_MODE:
        print("[WARNING] tiktoken not available, using rough estimates")
