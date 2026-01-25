"""Utility functions for token estimation and text extraction"""
import hashlib
from typing import List, Dict, Any, Union
from .config import TIKTOKEN_AVAILABLE

def estimate_tokens(text: str) -> int:
    """Estimate token count - use tiktoken if available, else rough estimate"""
    if TIKTOKEN_AVAILABLE:
        try:
            import tiktoken
            encoding = tiktoken.get_encoding("cl100k_base")
            return len(encoding.encode(text))
        except Exception:
            pass
    # Fallback: 1 token ≈ 4 characters
    return len(text) // 4


def extract_text_from_content(content: Union[str, List[Dict[str, Any]]]) -> str:
    """Extract text from string or multimodal content"""
    if isinstance(content, str):
        return content
    
    # Multimodal format
    text_parts = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") in ("text", "input_text") and "text" in item:
                text_parts.append(item["text"])
    
    return "\n".join(text_parts)


def get_conversation_id(messages: List[Dict]) -> str:
    """Generate conversation ID from first message content"""
    if len(messages) > 0:
        content = messages[0].get("content", "")
        content_text = extract_text_from_content(content)[:100]
        return hashlib.md5(content_text.encode()).hexdigest()
    return "default"


def total_tokens(messages: List[Dict], tools: List[Dict] = None) -> int:
    """Calculate total tokens for messages and tools with overhead"""
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
