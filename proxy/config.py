"""
Proxy configuration - imports from central settings.

This module exists for backwards compatibility and convenience.
All configuration is centralized in stack.settings.
"""
from stack.settings import (
    VLLM_URL,
    VISION_URL as VISION_API_URL,
    COMPRESSION_ENABLED,
    COMPRESSION_THRESHOLD,
    MAX_PROMPT_TOKENS,
    MODEL_MAX_CONTEXT,
    MODEL_TOOL_FORMAT,
    KEEP_RECENT_MESSAGES,
    COMPRESSION_RATE,
    DEBUG,
)

__all__ = [
    "VLLM_URL",
    "VISION_API_URL",
    "COMPRESSION_ENABLED",
    "COMPRESSION_THRESHOLD",
    "MAX_PROMPT_TOKENS",
    "MODEL_MAX_CONTEXT",
    "MODEL_TOOL_FORMAT",
    "KEEP_RECENT_MESSAGES",
    "COMPRESSION_RATE",
    "DEBUG",
]
