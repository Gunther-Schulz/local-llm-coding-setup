"""Tool parsing service implementation."""

from typing import List, Dict, Any, Optional
import json
import os
import re
import uuid

from stack.settings import DEBUG
from proxy.tool_parser import (
    _extract_braced_json,
    _make_tool_call,
    parse_qwen_tool_calls
)


class ToolService:
    """Tool parsing service for handling tool call transformations."""
    
    def __init__(self):
        """Initialize tool service."""
        pass
    
    def should_transform_tool_calls(self, vllm_sent: bool) -> bool:
        """Check if we should transform tool calls based on model and vLLM behavior.
        Reads MODEL_TOOL_FORMAT at call time so proxy launcher's env is respected."""
        if vllm_sent:
            return False  # vLLM already handled it
        
        fmt = os.getenv("MODEL_TOOL_FORMAT", "openai")
        return fmt in ("qwen2.5", "qwen3", "auto")
    
    def transform_qwen_response(self, response_data: Dict) -> Dict:
        """Transform non-streaming Qwen response to extract tool calls."""
        if not isinstance(response_data, dict):
            return response_data
        
        choices = response_data.get("choices", [])
        for choice in choices:
            message = choice.get("message", {})
            content = message.get("content", "")
            
            # Only transform if no tool_calls and content has tool calls
            if not message.get("tool_calls") and isinstance(content, str):
                tool_calls = parse_qwen_tool_calls(content)
                
                if tool_calls:
                    if DEBUG:
                        print(f"[DEBUG] Extracted {len(tool_calls)} tool call(s) from response")
                    
                    message["tool_calls"] = tool_calls
                    message["content"] = None
                    choice["finish_reason"] = "tool_calls"
        
        return response_data