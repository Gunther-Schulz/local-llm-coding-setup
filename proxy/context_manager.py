"""Context management using Cursor's strategy: summarization + sliding window."""
from typing import List, Dict, Any
import httpx

from stack.settings import DEBUG, BACKEND_URL


# Store old conversation history for retrieval
_conversation_archives: Dict[str, List[Dict]] = {}


async def summarize_conversation(messages: List[Dict]) -> str:
    """Use the LLM to summarize old conversation context."""
    # Build summarization prompt
    conversation_text = "\n\n".join([
        f"{msg.get('role', 'user')}: {_extract_content_preview(msg)}"
        for msg in messages
    ])
    
    summary_prompt = f"""Summarize this conversation history in 2-3 concise paragraphs. Focus on:
- What the user is working on
- Key decisions made
- Current state and goals
- Important context that should be remembered

Conversation:
{conversation_text}

Summary:"""

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{BACKEND_URL}/v1/chat/completions",
                json={
                    "model": "qwen3-30b-q2",
                    "messages": [{"role": "user", "content": summary_prompt}],
                    "max_tokens": 512,
                    "temperature": 0.3
                }
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]
    except Exception as e:
        if DEBUG:
            print(f"[WARNING] Summarization failed: {e}, using fallback")
        return f"[Previous conversation with {len(messages)} messages about code work]"


def _extract_content_preview(msg: Dict, max_chars: int = 200) -> str:
    """Extract preview of message content."""
    content = msg.get("content", "")
    
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))
        text = " ".join(text_parts)
    else:
        text = str(content)
    
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def condense_large_tool_response(msg: Dict) -> Dict:
    """Condense large tool responses - show preview, keep full text retrievable."""
    if msg.get("role") != "tool":
        return msg
    
    content = msg.get("content", "")
    if isinstance(content, list) and len(content) > 0:
        text_content = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
    else:
        text_content = str(content)
    
    # If tool response is large (>2000 chars), condense it
    if len(text_content) > 2000:
        preview = text_content[:500] + f"\n\n[... {len(text_content) - 500} more characters omitted ...]"
        
        condensed = msg.copy()
        if isinstance(content, list):
            condensed["content"] = [{"type": "text", "text": preview}]
        else:
            condensed["content"] = preview
        
        # Keep reference to full content for retrieval if needed
        condensed["_full_content_length"] = len(text_content)
        
        return condensed
    
    return msg


async def manage_context(
    messages: List[Dict],
    conversation_id: str,
    max_messages: int = 20
) -> List[Dict]:
    """
    Manage context using Cursor's strategy:
    1. Keep last N message exchanges (sliding window)
    2. Summarize older messages
    3. Archive old messages for retrieval
    4. Condense large tool responses
    """
    if len(messages) <= max_messages:
        # Small conversation, no management needed
        return messages
    
    # Separate system prompt from conversation
    system_prompt = None
    conversation_messages = messages
    
    if messages and messages[0].get("role") == "system":
        system_prompt = messages[0]
        conversation_messages = messages[1:]
    
    # Split: old messages to summarize vs recent to keep
    split_point = len(conversation_messages) - max_messages
    old_messages = conversation_messages[:split_point]
    recent_messages = conversation_messages[split_point:]
    
    if DEBUG:
        print(f"[DEBUG] Context management: {len(old_messages)} old, {len(recent_messages)} recent")
    
    # Archive old messages for retrieval
    if conversation_id not in _conversation_archives:
        _conversation_archives[conversation_id] = []
    _conversation_archives[conversation_id].extend(old_messages)
    
    if DEBUG:
        print(f"[DEBUG] Archived {len(old_messages)} messages (total archive: {len(_conversation_archives[conversation_id])})")
    
    # Summarize old context
    summary = await summarize_conversation(old_messages)
    
    if DEBUG:
        print(f"[DEBUG] Generated summary: {summary[:100]}...")
    
    summary_message = {
        "role": "user",
        "content": f"[Previous conversation summary]:\n{summary}\n\n[Continuing conversation...]"
    }
    
    # Condense large tool responses in recent messages
    condensed_recent = [condense_large_tool_response(msg) for msg in recent_messages]
    
    # Build final context
    final_messages = []
    if system_prompt:
        final_messages.append(system_prompt)
    final_messages.append(summary_message)
    final_messages.extend(condensed_recent)
    
    if DEBUG:
        print(f"[DEBUG] Final context: {len(final_messages)} messages (1 summary + {len(condensed_recent)} recent)")
    
    return final_messages
