"""Tool call parsing and transformation for Qwen models."""
import json
import os
import re
import uuid
from typing import List, Dict, Optional

from stack.settings import DEBUG


def _extract_braced_json(s: str, start: int) -> tuple[Optional[str], int]:
    """Extract a single {...} JSON object from s starting at start; return (json_str, end_pos)."""
    i = s.find("{", start)
    if i < 0:
        return None, start
    depth = 0
    j = i
    while j < len(s):
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
            if depth == 0:
                return s[i : j + 1], j + 1
        elif s[j] == '"' and (j == 0 or s[j - 1] != "\\"):
            j += 1
            while j < len(s) and (s[j] != '"' or s[j - 1] == "\\"):
                j += 1
        j += 1
    return None, start


def _make_tool_call(name: str, arguments: dict) -> Dict:
    """Build OpenAI-compatible tool call dict."""
    return {
        "id": f"call_{uuid.uuid4().hex[:24]}",
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments) if isinstance(arguments, dict) else json.dumps({})
        }
    }


def parse_qwen_tool_calls(content: str) -> Optional[List[Dict]]:
    """
    Parse tool calls from XML or JSON format.
    
    Formats supported:
    1. Cursor-style: <read><file>path</file></read>, <function=Read><file>path</file></function>
    2. Qwen XML: <function>{"name":"...", "arguments":{...}}</function>
    3. Markdown: ```json\n{"name":"...", "arguments":{...}}\n```
    
    Returns OpenAI-compatible tool_calls array.
    """
    tool_calls = []
    seen: set = set()  # (name, args_str) to avoid duplicates

    # Cursor-style: <read><file>path</file></read> or <tool_name><file>path</file></tool_name>
    for m in re.finditer(r"<(\w+)>\s*<file>(.*?)</file>\s*</\1>", content, re.DOTALL | re.IGNORECASE):
        tool_name = m.group(1).strip()
        path = m.group(2).strip()
        if not path or not tool_name:
            continue
        name = tool_name[0].upper() + tool_name[1:].lower() if tool_name else ""
        key = (name, path)
        if key not in seen:
            seen.add(key)
            tool_calls.append(_make_tool_call(name, {"path": path}))

    # <function=Read> or <function name="Read"> with <file>path</file> or <parameter=path>...</parameter> inside
    for m in re.finditer(r"<function\s*=?\s*(\w+)>(.*?)</function>", content, re.DOTALL | re.IGNORECASE):
        name = m.group(1).strip()
        if not name:
            continue
        name = name[0].upper() + name[1:].lower()
        inner = m.group(2)
        path = None
        file_m = re.search(r"<file>(.*?)</file>", inner, re.DOTALL | re.IGNORECASE)
        if file_m:
            path = file_m.group(1).strip()
        if not path:
            # <parameter=path>value</parameter> or <parameter name="path">value</parameter>
            param_m = re.search(r"<parameter\s+(?:name=[\"']?path[\"']?|=\s*[\"']?path[\"']?)\s*>(.*?)</parameter>", inner, re.DOTALL | re.IGNORECASE)
            if not param_m:
                param_m = re.search(r"<parameter=path>(.*?)</parameter>", inner, re.DOTALL | re.IGNORECASE)
            if param_m:
                path = param_m.group(1).strip()
        if path:
            key = (name, path)
            if key not in seen:
                seen.add(key)
                tool_calls.append(_make_tool_call(name, {"path": path}))

    # Qwen XML: <function>{"name":"...", "arguments":{...}}</function>
    for tag in ("function", "tool_call"):
        pattern = re.compile(
            rf"<{tag}\s*>(.*?)</{tag}>",
            re.DOTALL,
        )
        for m in pattern.finditer(content):
            inner = m.group(1).strip()
            json_str, _ = _extract_braced_json(inner, 0)
            if not json_str:
                continue
            try:
                tool_data = json.loads(json_str)
                name = tool_data.get("name", "")
                args = tool_data.get("arguments", {})
                key = (name, json.dumps(args, sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    tool_calls.append(_make_tool_call(name, args))
            except json.JSONDecodeError:
                if DEBUG:
                    print(f"[DEBUG] Failed to parse XML tool call: {json_str[:100]}")
    
    # Pattern 2: JSON in markdown code blocks
    for m in re.finditer(r"```(?:json)?\s*\n", content):
        json_str, end_pos = _extract_braced_json(content, m.end())
        if not json_str:
            continue
        try:
            tool_data = json.loads(json_str)
            if "name" in tool_data:
                name = tool_data.get("name", "")
                args = tool_data.get("arguments", {})
                key = (name, json.dumps(args, sort_keys=True))
                if key not in seen:
                    seen.add(key)
                    tool_calls.append(_make_tool_call(name, args))
        except json.JSONDecodeError:
            if DEBUG:
                print(f"[DEBUG] Failed to parse markdown tool call: {json_str[:100]}")
    
    return tool_calls if tool_calls else None


def should_transform_tool_calls(backend_sent_tool_calls: bool) -> bool:
    """Check if we should transform tool calls based on model and backend behavior.
    Reads MODEL_TOOL_FORMAT at call time so proxy launcher's env is respected."""
    if backend_sent_tool_calls:
        return False  # Backend already sent tool_calls in stream
    
    fmt = os.getenv("MODEL_TOOL_FORMAT", "openai")
    return fmt in ("qwen2.5", "qwen3", "auto")


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
