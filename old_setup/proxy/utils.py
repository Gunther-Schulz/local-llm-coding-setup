"""Utility functions for token estimation and text extraction."""
import hashlib
from typing import List, Dict, Any, Union

# Load Qwen tokenizer once at module startup
_TOKENIZER = None

try:
    from transformers import AutoTokenizer
    _TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-14B-Instruct")
    print("[INFO] Loaded Qwen tokenizer for accurate token counting")
except Exception as e:
    print(f"[WARNING] Could not load Qwen tokenizer: {e}")
    print("[WARNING] Falling back to rough estimation (1 token ≈ 3 chars)")


def estimate_tokens(text: str) -> int:
    """Estimate token count using Qwen's actual tokenizer."""
    if _TOKENIZER is not None:
        try:
            return len(_TOKENIZER.encode(text, add_special_tokens=False))
        except Exception as e:
            print(f"[WARNING] Tokenizer error: {e}, using fallback")
    
    # Fallback: Qwen tokens are roughly 1 token per 3 characters
    return len(text) // 3


def extract_text_from_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract text from string or multimodal content."""
    if isinstance(content, str):
        return content
    
    # Multimodal format
    text_parts = []
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text") and "text" in item:
                    text_parts.append(item["text"])
    
    return "\n".join(text_parts)


def get_conversation_id(messages: List[Dict]) -> str:
    """Generate conversation ID from first non-system message content.
    Avoids keying off the system message so conversations with the same
    system prompt don't collide in the compressed store."""
    for msg in messages:
        if msg.get("role") == "system":
            continue
        content = msg.get("content", "")
        content_text = extract_text_from_content(content)[:100]
        if content_text.strip():
            return hashlib.md5(content_text.encode()).hexdigest()
    # Fallback: only system message(s) or empty – use first message
    if len(messages) > 0:
        content = messages[0].get("content", "")
        content_text = extract_text_from_content(content)[:100]
        return hashlib.md5(content_text.encode()).hexdigest()
    return "default"


def total_tokens(messages: List[Dict], tools: List[Dict] = None) -> int:
    """Calculate total tokens for messages and tools with overhead."""
    import json
    
    # Message tokens
    text_parts = []
    for m in messages:
        content = m.get("content", "")
        text_parts.append(extract_text_from_content(content))
        
        # Add overhead for message structure (role, formatting, etc.)
        # Each message has ~10 tokens of overhead
        text_parts.append(" " * 40)  # ~10 tokens overhead per message
    
    message_text = " ".join(text_parts)
    message_tokens = estimate_tokens(message_text)
    
    # Tool tokens (tools add significant overhead)
    tool_tokens = 0
    if tools:
        tools_json = json.dumps(tools)
        tool_tokens = estimate_tokens(tools_json)
        # Tools have additional template overhead
        tool_tokens = int(tool_tokens * 1.2)  # Add 20% overhead for tool template
    
    return message_tokens + tool_tokens
