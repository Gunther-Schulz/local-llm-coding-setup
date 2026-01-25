"""Message compression logic for context management."""
from typing import List, Dict, Optional

from stack.settings import (
    COMPRESSION_THRESHOLD,
    COMPRESSION_ENABLED,
    COMPRESSION_RATE,
    KEEP_RECENT_MESSAGES,
    DEBUG
)
from proxy.utils import extract_text_from_content, estimate_tokens

# Lazy-loaded compressor (initialized on first use)
_compressor: Optional[object] = None

# Caches (only used if server-side caching is enabled - not used with Cursor)
conversation_cache: Dict[str, List[Dict]] = {}


def get_compressor():
    """Lazy-load compressor on CPU to avoid GPU OOM."""
    global _compressor
    if _compressor is None:
        try:
            from llmlingua import PromptCompressor
            _compressor = PromptCompressor(
                model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                use_llmlingua2=True,
                device_map="cpu"  # Force CPU to avoid conflict with vLLM on GPU
            )
            if DEBUG:
                print("[DEBUG] Compression model loaded on CPU")
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Failed to load compression model: {e}")
            _compressor = False  # Mark as unavailable
    return _compressor if _compressor is not False else None


def compress_messages(messages: List[Dict], keep_recent: int = KEEP_RECENT_MESSAGES) -> List[Dict]:
    """Compress old messages, preserve recent ones and tool calls."""
    if len(messages) <= keep_recent:
        return messages
    
    compressor = get_compressor()
    if not compressor:
        # Compression unavailable, just truncate old messages
        if DEBUG:
            print("[DEBUG] Compression unavailable, truncating to recent messages")
        return messages[-keep_recent:]
    
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
        
        # Truncate if combined text is too large for compression model
        # llmlingua's internal model has limits, so we pre-truncate if needed
        from proxy.utils import estimate_tokens
        text_tokens = estimate_tokens(combined_text)
        if text_tokens > 15000:  # Safe limit for compression model
            # Take middle portion to preserve some history
            words = combined_text.split()
            keep_words = int(len(words) * 0.6)  # Keep 60% of words
            combined_text = " ".join(words[-keep_words:])
            if DEBUG:
                print(f"[DEBUG] Pre-truncated text for compression: {text_tokens} -> ~{estimate_tokens(combined_text)} tokens")
        
        try:
            # Compression rate from central config
            compressed_result = compressor.compress_prompt(combined_text, rate=COMPRESSION_RATE)
            compressed_text = compressed_result.get("compressed_prompt", combined_text)
            
            if DEBUG:
                orig_tokens = estimate_tokens(combined_text)
                comp_tokens = estimate_tokens(compressed_text)
                print(f"[DEBUG] Compressed {orig_tokens} → {comp_tokens} tokens ({comp_tokens*100//orig_tokens}%)")
            
            # Use 'user' role instead of 'system' to avoid vLLM validation issues
            summary_message = {
                "role": "user",
                "content": f"[Previous conversation summary]: {compressed_text}"
            }
            
            return [summary_message] + non_compressible + recent_messages
        except Exception as e:
            if DEBUG:
                print(f"[DEBUG] Compression failed: {e}")
            # Fallback: just use non-compressible + recent
            return non_compressible + recent_messages
    
    return non_compressible + recent_messages


def manage_conversation_history(
    conversation_id: str,
    incoming_messages: List[Dict]
) -> List[Dict]:
    """Manage conversation history with optional compression."""
    if not COMPRESSION_ENABLED:
        return incoming_messages
    
    if conversation_id not in conversation_cache:
        conversation_cache[conversation_id] = []
    
    conversation_cache[conversation_id].extend(incoming_messages)
    
    # Estimate tokens
    total_text = " ".join([
        extract_text_from_content(msg.get("content", ""))
        for msg in conversation_cache[conversation_id]
    ])
    estimated_tokens = estimate_tokens(total_text)
    
    if DEBUG:
        print(f"[DEBUG] Conversation {conversation_id}: {estimated_tokens} tokens, {len(conversation_cache[conversation_id])} messages")
    
    # Compress if over threshold
    if estimated_tokens > COMPRESSION_THRESHOLD:
        if DEBUG:
            print(f"[DEBUG] Triggering compression (threshold: {COMPRESSION_THRESHOLD})")
        conversation_cache[conversation_id] = compress_messages(
            conversation_cache[conversation_id]
        )
    
    return conversation_cache[conversation_id]
