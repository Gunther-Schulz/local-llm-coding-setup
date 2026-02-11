"""Context management service: Cursor-style compression on overflow + tool condense."""

from typing import List, Dict, Any, Optional

from stack.settings import DEBUG, TOOL_RESPONSE_NO_CONDENSE_PATHS
from proxy.context_manager import (
    compress_cursor_style as _compress_cursor_style,
    _extract_content_preview,
    _build_tool_call_id_to_path,
    _path_matches_no_condense,
    condense_large_tool_response,
    condense_tool_responses_with_context as _condense_tool_responses_with_context,
)


class ContextService:
    """Context management service: Cursor-style compression on overflow + tool condense."""
    
    def __init__(self):
        """Initialize context service."""
        pass

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

    async def compress_cursor_style(
        self,
        messages: List[Dict],
        conversation_id: str,
        recent_count: int = 6,
        model: Optional[str] = None,
    ) -> List[Dict]:
        """Cursor-style compression: structured summary + last N messages. Use when prompt would exceed backend limit. model: use this for summarization (e.g. request.model); else fallback from settings."""
        return await _compress_cursor_style(messages, conversation_id, recent_count=recent_count, model=model)