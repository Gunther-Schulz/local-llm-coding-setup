"""Streaming response handling with tool call transformation."""
import json
import time
import uuid
from typing import Generator

from stack.settings import DEBUG
from proxy.tool_parser import (
    parse_qwen_tool_calls,
    should_transform_tool_calls,
    generate_tool_call_chunks
)


class StreamState:
    """Track state during streaming."""
    def __init__(self, model_name: str):
        self.chunk_count = 0
        self.accumulated_content = ""
        self.finish_reason = None
        self.first_chunk_id = None
        self.vllm_sent_tool_calls = False
        self.model_name = model_name
        self.created_time = int(time.time())
    
    def get_chunk_id(self) -> str:
        """Get or generate chunk ID."""
        return self.first_chunk_id or f"chatcmpl-{uuid.uuid4().hex[:16]}"


def process_stream_end(state: StreamState) -> Generator[str, None, None]:
    """Process end of stream and send tool calls if needed."""
    tool_calls = None
    
    # Try to extract tool calls if we should transform
    if state.accumulated_content and should_transform_tool_calls(state.vllm_sent_tool_calls):
        tool_calls = parse_qwen_tool_calls(state.accumulated_content)
        
        if DEBUG and tool_calls:
            print(f"[DEBUG] Extracted {len(tool_calls)} tool call(s) from stream")
    
    # Send tool calls if found
    if tool_calls:
        yield from generate_tool_call_chunks(
            tool_calls, state.get_chunk_id(), 
            state.model_name, state.created_time
        )
    
    # Log summary
    if DEBUG:
        print(f"[DEBUG] ===== STREAM SUMMARY =====")
        print(f"[DEBUG]   Chunks: {state.chunk_count}")
        print(f"[DEBUG]   Content: {len(state.accumulated_content)} chars")
        print(f"[DEBUG]   Tool calls: {len(tool_calls) if tool_calls else 0}")
        print(f"[DEBUG]   Finish: {'tool_calls' if tool_calls else state.finish_reason}")
        print(f"[DEBUG] ==========================")
    
    # Always send [DONE]
    yield "data: [DONE]\n\n"


def stream_with_tool_transform(upstream, model_name: str) -> Generator[str, None, None]:
    """
    Stream SSE events from vLLM and transform tool calls.
    
    This is the main streaming handler that:
    1. Passes through all chunks as-is
    2. Accumulates content for tool call detection
    3. At stream end, extracts and sends tool calls if needed
    """
    state = StreamState(model_name)
    
    if DEBUG:
        print(f"[DEBUG] ===== STREAMING START =====")
    
    try:
        for chunk in upstream.iter_content(chunk_size=None):
            if not chunk:
                continue
            
            chunk_str = chunk.decode('utf-8') if isinstance(chunk, bytes) else chunk
            
            for line in chunk_str.split('\n'):
                # Pass through non-data lines
                if not line.strip() or not line.startswith('data: '):
                    if line.strip():
                        yield f"{line}\n"
                    continue
                
                data_str = line[6:].strip()
                
                # Handle stream end
                if data_str == "[DONE]":
                    if DEBUG:
                        print(f"[DEBUG] Stream completed with [DONE]")
                    
                    yield from process_stream_end(state)
                    return
                
                # Parse chunk data
                try:
                    chunk_data = json.loads(data_str)
                    state.chunk_count += 1
                    
                    if not state.first_chunk_id:
                        state.first_chunk_id = chunk_data.get("id")
                    
                    # Extract metadata
                    for choice in chunk_data.get("choices", []):
                        delta = choice.get("delta", {})
                        
                        if choice.get("finish_reason"):
                            state.finish_reason = choice["finish_reason"]
                        
                        if "tool_calls" in delta and delta["tool_calls"]:
                            state.vllm_sent_tool_calls = True
                            if DEBUG:
                                print(f"[DEBUG] vLLM sent tool_calls (native parser worked!)")
                        
                        if "content" in delta and delta["content"]:
                            state.accumulated_content += delta["content"]
                    
                    # Pass through as-is
                    yield f"{line}\n"
                    
                except json.JSONDecodeError:
                    yield f"{line}\n"
        
        # Stream ended without [DONE]
        if DEBUG:
            print(f"[DEBUG] ⚠ Stream ended without [DONE]")
        
        yield from process_stream_end(state)
    
    finally:
        upstream.close()
