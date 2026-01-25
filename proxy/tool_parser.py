"""Tool call parsing and transformation for Qwen models."""
import json
import re
import uuid
from typing import List, Dict, Optional

from stack.settings import DEBUG, MODEL_TOOL_FORMAT


def parse_qwen_tool_calls(content: str) -> Optional[List[Dict]]:
    """
    Parse Qwen tool calls from XML or JSON-in-markdown format.
    
    Formats supported:
    1. XML: <function>{"name":"...", "arguments":{...}}</function>
    2. Markdown: ```json\n{"name":"...", "arguments":{...}}\n```
    
    Returns OpenAI-compatible tool_calls array.
    """
    tool_calls = []
    
    # Pattern 1: XML tags
    xml_pattern = r'<(?:function|tool_call)>\s*(\{[^}]*\})\s*</(?:function|tool_call)>'
    xml_matches = re.findall(xml_pattern, content, re.DOTALL)
    
    for match in xml_matches:
        try:
            tool_data = json.loads(match)
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:24]}",
                "type": "function",
                "function": {
                    "name": tool_data.get("name", ""),
                    "arguments": json.dumps(tool_data.get("arguments", {}))
                }
            }
            tool_calls.append(tool_call)
        except json.JSONDecodeError:
            if DEBUG:
                print(f"[DEBUG] Failed to parse XML tool call: {match[:100]}")
    
    # Pattern 2: JSON in markdown
    markdown_pattern = r'```(?:json)?\s*\n(\{[^`]+\})\s*\n```'
    markdown_matches = re.findall(markdown_pattern, content, re.DOTALL)
    
    for match in markdown_matches:
        try:
            tool_data = json.loads(match)
            if "name" in tool_data:  # Looks like a tool call
                tool_call = {
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": tool_data.get("name", ""),
                        "arguments": json.dumps(tool_data.get("arguments", {}))
                    }
                }
                tool_calls.append(tool_call)
        except json.JSONDecodeError:
            if DEBUG:
                print(f"[DEBUG] Failed to parse markdown tool call: {match[:100]}")
    
    return tool_calls if tool_calls else None


def should_transform_tool_calls(vllm_sent: bool) -> bool:
    """Check if we should transform tool calls based on model and vLLM behavior."""
    if vllm_sent:
        return False  # vLLM already handled it
    
    return MODEL_TOOL_FORMAT in ("qwen2.5", "qwen3", "auto")


def transform_qwen_response(response_data: Dict) -> Dict:
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


def generate_tool_call_chunks(tool_calls: List[Dict], chunk_id: str, model_name: str, created_time: int):
    """Generate SSE chunks for tool calls in OpenAI format."""
    # Send each tool call as delta
    for i, tool_call in enumerate(tool_calls):
        tool_chunk = {
            "id": chunk_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {
                    "tool_calls": [{
                        "index": i,
                        "id": tool_call["id"],
                        "type": "function",
                        "function": {
                            "name": tool_call["function"]["name"],
                            "arguments": tool_call["function"]["arguments"]
                        }
                    }]
                },
                "finish_reason": None
            }]
        }
        yield f"data: {json.dumps(tool_chunk)}\n\n"
    
    # Send final chunk with finish_reason
    final_chunk = {
        "id": chunk_id,
        "object": "chat.completion.chunk",
        "created": created_time,
        "model": model_name,
        "choices": [{
            "index": 0,
            "delta": {},
            "finish_reason": "tool_calls"
        }]
    }
    yield f"data: {json.dumps(final_chunk)}\n\n"
