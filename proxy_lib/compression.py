"""Message compression logic"""
from typing import List, Dict, Optional
from .config import COMPRESSION_THRESHOLD, KEEP_RECENT_MESSAGES
from .utils import extract_text_from_content, estimate_tokens

# Lazy-loaded compressor (initialized on first use)
_compressor: Optional[object] = None

# Caches
conversation_cache: Dict[str, List[Dict]] = {}
rolling_summaries: Dict[str, str] = {}
compressed_cache: Dict[str, str] = {}


def get_compressor():
    """Lazy-load compressor on CPU to avoid GPU OOM"""
    global _compressor
    if _compressor is None:
        from llmlingua import PromptCompressor
        _compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="cpu"  # Force CPU to avoid conflict with vLLM on GPU
        )
    return _compressor


def compress_messages(messages: List[Dict], keep_recent: int = KEEP_RECENT_MESSAGES) -> List[Dict]:
    """Compress old messages, preserve recent ones and tool calls"""
    if len(messages) <= keep_recent:
        return messages
    
    old_messages = messages[:-keep_recent]
    recent_messages = messages[-keep_recent:]
    
    # Separate compressible messages from tool calls/responses
    compressible = []
    non_compressible = []
    
    for msg in old_messages:
        role = msg.get("role")
        if role in ("tool", "function") or msg.get("tool_calls") or msg.get("tool_call_id"):
            non_compressible.append(msg)
        else:
            compressible.append(msg)
    
    # Compress the compressible messages
    if compressible:
        combined_text = "\n\n".join([
            f"{m.get('role', 'user')}: {extract_text_from_content(m.get('content', ''))}"
            for m in compressible
        ])
        
        compressor = get_compressor()
        compressed_result = compressor.compress_prompt(combined_text, rate=0.5)
        compressed_text = compressed_result.get("compressed_prompt", combined_text)
        
        # Use 'user' role instead of 'system' to avoid vLLM validation issues
        summary_message = {
            "role": "user",
            "content": f"[Previous conversation summary]: {compressed_text}"
        }
        
        return [summary_message] + non_compressible + recent_messages
    
    return non_compressible + recent_messages


def manage_conversation_history(
    conversation_id: str,
    incoming_messages: List[Dict]
) -> List[Dict]:
    """Manage conversation history with compression"""
    if conversation_id not in conversation_cache:
        conversation_cache[conversation_id] = []
    
    conversation_cache[conversation_id].extend(incoming_messages)
    
    # Estimate tokens
    total_text = " ".join([
        extract_text_from_content(msg.get("content", ""))
        for msg in conversation_cache[conversation_id]
    ])
    estimated_tokens = estimate_tokens(total_text)
    
    # Compress if over threshold
    if estimated_tokens > COMPRESSION_THRESHOLD:
        conversation_cache[conversation_id] = compress_messages(
            conversation_cache[conversation_id]
        )
    
    return conversation_cache[conversation_id]
