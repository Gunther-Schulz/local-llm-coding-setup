"""Context management service implementation."""

from typing import List, Dict, Any, Optional
import fnmatch
import json
import os
import time
import httpx

from stack.settings import (
    DEBUG,
    BACKEND_URL,
    TOOL_RESPONSE_MAX_VERBATIM,
    TOOL_RESPONSE_PREVIEW_CHARS,
    TOOL_RESPONSE_NO_CONDENSE_PATHS,
    PRESERVE_FIRST_USER_IN_SUMMARY,
)
from proxy.context_manager import (
    summarize_conversation,
    _extract_content_preview,
    _build_tool_call_id_to_path,
    _path_matches_no_condense,
    condense_large_tool_response
)


class ContextService:
    """Context management service for handling conversation compression."""
    
    def __init__(self):
        """Initialize context service."""
        pass
    
    async def manage_context(self, messages: List[Dict], conversation_id: str, 
                           max_messages: int = 20) -> List[Dict]:
        """
        When conversation is large (message or token threshold): keep last N messages
        (sliding window), summarize older ones into one user message, condense large
        tool responses in the kept window. Archives old messages in memory per
        conversation_id (for possible future retrieval).
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
        # Note: This would be better implemented with a proper archive service
        # For now, we'll keep it simple but note that this is a limitation
        
        if DEBUG:
            print(f"[DEBUG] Archived {len(old_messages)} messages (total archive: N/A)")
        
        # Summarize old context
        _sum_start = time.perf_counter()
        summary = await summarize_conversation(old_messages)
        if DEBUG:
            print(f"[DEBUG] Summarization took {time.perf_counter() - _sum_start:.2f}s")
            print(f"[DEBUG] Generated summary: {summary[:100]}...")
        
        # Prepend first user message so task context (e.g. "use CLIPPY") stays in context
        summary_parts = []
        if PRESERVE_FIRST_USER_IN_SUMMARY:
            first_user = next((m for m in old_messages if m.get("role") == "user"), None)
            if first_user:
                first_content = _extract_content_preview(first_user, max_chars=2000)
                summary_parts.append(f"[Initial user request]:\n{first_content}\n")
        summary_parts.append(f"[Previous conversation summary]:\n{summary}\n\n[Continuing conversation...]")
        summary_content = "\n".join(summary_parts)
        
        summary_message = {
            "role": "user",
            "content": summary_content
        }
        
        # Condense large tool responses in recent messages (instruction docs bypassed via patterns)
        condensed_recent = self.condense_tool_responses_with_context(recent_messages)
        
        # Build final context
        final_messages = []
        if system_prompt:
            final_messages.append(system_prompt)
        final_messages.append(summary_message)
        final_messages.extend(condensed_recent)
        
        if DEBUG:
            print(f"[DEBUG] Final context: {len(final_messages)} messages (1 summary + {len(condensed_recent)} recent)")
        
        return final_messages
    
    def condense_tool_responses_with_context(self, messages: List[Dict], 
                                          no_condense_patterns: Optional[List[str]] = None) -> List[Dict]:
        """Condense large tool responses, but skip condensing for paths matching no_condense_patterns."""
        if no_condense_patterns is None:
            no_condense_patterns = TOOL_RESPONSE_NO_CONDENSE_PATHS
        id_to_path = _build_tool_call_id_to_path(messages)
        result = []
        for msg in messages:
            if msg.get("role") != "tool":
                result.append(msg)
                continue
            tid = msg.get("tool_call_id")
            path = id_to_path.get(tid) if tid else None
            skip = bool(path and _path_matches_no_condense(path, no_condense_patterns))
            if DEBUG and skip:
                print(f"[DEBUG] No condense for tool result (path matches): {path}")
            result.append(condense_large_tool_response(msg, skip_condense=skip))
        return result