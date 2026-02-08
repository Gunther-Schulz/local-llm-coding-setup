"""
Central configuration for the entire system.

All configurable settings for proxy, vision, and llama-server backend are defined here.
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

# LLM backend URL (llama-server); proxy and clients use this
BACKEND_URL = os.getenv("BACKEND_URL", f"http://localhost:{VLLM_PORT}")

PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", "8002"))

VISION_HOST = os.getenv("VISION_HOST", "0.0.0.0")
VISION_PORT = int(os.getenv("VISION_PORT", "8004"))
VISION_URL = os.getenv("VISION_API_URL", f"http://localhost:{VISION_PORT}")


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
# Compression Settings (Cursor-style: trigger on overflow only)
# ============================================================================
# When 1: condense tool responses on every request; when prompt would exceed backend limit (413),
# apply Cursor-style compression (structured summary + last N messages) instead of returning 413.
# When 0: pass-through; return 413 when over limit (e.g. for Cursor Cloud which compresses on 413).
COMPRESSION_ENABLED = os.getenv("COMPRESSION_ENABLED", "0") == "1"

# Cursor-style: how many recent messages to keep verbatim after the [Previous conversation summary].
CONTEXT_WINDOW_SIZE = int(os.getenv("CONTEXT_WINDOW_SIZE", "6"))
# Tool response condensing (messages with role=tool longer than this get preview only)
TOOL_RESPONSE_MAX_VERBATIM = int(os.getenv("TOOL_RESPONSE_MAX_VERBATIM", "2000"))
TOOL_RESPONSE_PREVIEW_CHARS = int(os.getenv("TOOL_RESPONSE_PREVIEW_CHARS", "500"))
# Comma-separated path patterns for which tool responses are never condensed (fnmatch).
_TOOL_NO_CONDENSE = os.getenv("TOOL_RESPONSE_NO_CONDENSE_PATHS", "")
TOOL_RESPONSE_NO_CONDENSE_PATHS = [p.strip() for p in _TOOL_NO_CONDENSE.split(",") if p.strip()]
# Prepend the first user message to the summary so task context (e.g. "use CLIPPY") is kept
PRESERVE_FIRST_USER_IN_SUMMARY = os.getenv("PRESERVE_FIRST_USER_IN_SUMMARY", "1") == "1"

# Summarization model for Cursor-style compression (when COMPRESSION_ENABLED=1).
# When the proxy passes the request model (the active chat model), that is used. This is only the fallback when no model is passed (e.g. internal call). Must be a model ID the backend exposes.
COMPRESSION_SUMMARY_MODEL = os.getenv("COMPRESSION_SUMMARY_MODEL", "qwen3-30b-q2")
# Timeout in seconds for the summarization LLM call.
COMPRESSION_SUMMARY_TIMEOUT = int(os.getenv("COMPRESSION_SUMMARY_TIMEOUT", "60"))
# Fuzzy section match: minimum rapidfuzz ratio (0–100) to accept a section line. Used when exact header match fails.
COMPRESSION_SECTION_FUZZ_RATIO = int(os.getenv("COMPRESSION_SECTION_FUZZ_RATIO", "60"))

# Virtual tool (search_compressed_conversation): max compressed conversations to keep in memory.
# 0 or negative = unlimited (full history). Positive = FIFO eviction after that many.
COMPRESSED_STORE_MAX_CONVERSATIONS = int(os.getenv("COMPRESSED_STORE_MAX_CONVERSATIONS", "0"))
# Caps on virtual tool results (0 = no cap / unlimited)
COMPRESSED_STORE_RESULT_MAX_CHARS = int(os.getenv("COMPRESSED_STORE_RESULT_MAX_CHARS", "4000"))
COMPRESSED_STORE_SEARCH_TOP_K = int(os.getenv("COMPRESSED_STORE_SEARCH_TOP_K", "5"))
COMPRESSED_STORE_SEARCH_MAX_CHARS = int(os.getenv("COMPRESSED_STORE_SEARCH_MAX_CHARS", "3500"))


# ============================================================================
# System message injection (for clients that don't send one, e.g. Continue)
# ============================================================================
# When 1: replace/prepend with Cursor-style system message from SYSTEM_MESSAGE_FILE.
# When 0 (default): do not inject; client provides system message (e.g. Cursor Cloud).
# Set to 1 when using Continue or other clients that don't send a system message.
INJECT_SYSTEM_MESSAGE = os.getenv("INJECT_SYSTEM_MESSAGE", "0") == "1"

SYSTEM_MESSAGE_FILE = os.getenv("SYSTEM_MESSAGE_FILE", "config/system_message.txt")


def _load_system_message_text() -> str:
    """Load system message from SYSTEM_MESSAGE_FILE (project root relative). Empty if missing or unset."""
    root = Path(__file__).resolve().parents[1]
    path = root / SYSTEM_MESSAGE_FILE if not os.path.isabs(SYSTEM_MESSAGE_FILE) else Path(SYSTEM_MESSAGE_FILE)
    try:
        if path.exists():
            return path.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return ""


SYSTEM_MESSAGE_TEXT = _load_system_message_text()


# ============================================================================
# Tool Calling
# ============================================================================

# Tool parser format: "openai", "qwen2.5", "qwen3", "auto"
MODEL_TOOL_FORMAT = os.getenv("MODEL_TOOL_FORMAT", "openai")

# When 1: prepend a short capability reminder to the system message so the model uses
# conversation context and tools instead of claiming it cannot (recall history, WebSearch, edits).
INJECT_CAPABILITY_REMINDER = os.getenv("INJECT_CAPABILITY_REMINDER", "1") == "1"

_CAPABILITY_REMINDER_DEFAULT = (
    "\n\n<capability_reminder>\n"
    "In this session you have (1) the full conversation in the messages above—use it to recall or refer to earlier messages; "
    "(2) the tools listed in this request—use them when appropriate (e.g. WebSearch for current info, StrReplace/Write for file changes) rather than only advising. "
    "Do not claim you lack history or capabilities that are present in this request.\n"
    "</capability_reminder>\n"
)


def _load_capability_reminder_text() -> str:
    """Load capability reminder from config/capability_reminder.txt, or return default if missing."""
    root = Path(__file__).resolve().parents[1]
    path = root / "config" / "capability_reminder.txt"
    try:
        if path.exists():
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return "\n\n" + text + "\n"
    except OSError:
        pass
    return _CAPABILITY_REMINDER_DEFAULT


CAPABILITY_REMINDER_TEXT = _load_capability_reminder_text()


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
    """Get the effective context limit for the current model. Reads os.environ at call time
    so proxy/llm launchers that set MODEL_EXTENDED_CONTEXT after importing settings are respected."""
    extended_ctx = os.getenv("MODEL_EXTENDED_CONTEXT")
    if extended_ctx:
        return int(extended_ctx)
    return int(os.getenv("MODEL_MAX_CONTEXT", str(MODEL_MAX_CONTEXT)))


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
    print(f"Proxy:       {PROXY_HOST}:{PROXY_PORT}")
    print(f"Vision:      {VISION_URL}")
    print(f"Context:     {MODEL_MAX_CONTEXT} tokens")
    print(f"Max Prompt:  {MAX_PROMPT_TOKENS} tokens")
    print(f"Context Mgmt: Cursor-style (trigger on overflow only)")
    print(f"  Compression:  {'On' if COMPRESSION_ENABLED else 'Off'} (compress when prompt would exceed limit)")
    print(f"  Window Size:  {CONTEXT_WINDOW_SIZE} recent messages after summary")
    print(f"  Compressed store: {'unlimited' if COMPRESSED_STORE_MAX_CONVERSATIONS <= 0 else f'max {COMPRESSED_STORE_MAX_CONVERSATIONS} conversations'}")
    print(f"  Virtual tool caps: result={COMPRESSED_STORE_RESULT_MAX_CHARS or 'none'}, search_top_k={COMPRESSED_STORE_SEARCH_TOP_K or 'none'}, search_max_chars={COMPRESSED_STORE_SEARCH_MAX_CHARS or 'none'}")
    print(f"Debug:       {'On' if DEBUG else 'Off'}")
    print("=" * 70)
