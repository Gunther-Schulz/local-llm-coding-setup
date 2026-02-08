"""Streaming response handling with tool call transformation."""
import json
import time
import uuid
from typing import Generator, Optional

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
        self.tool_calls_sent = False
        self.tool_call_count = 0
        self.model_name = model_name
        self.created_time = int(time.time())
    
    def get_chunk_id(self) -> str:
        """Get or generate chunk ID."""
        return self.first_chunk_id or f"chatcmpl-{uuid.uuid4().hex[:16]}"


def process_stream_end(
    state: StreamState,
    request_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """Process end of stream and send tool calls if needed."""
    tool_calls = None

    # Try to extract tool calls if we should transform
    if state.accumulated_content and should_transform_tool_calls(state.tool_calls_sent):
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
        rid = request_id or "n/a"
        cid = (conversation_id[:12] + "..") if conversation_id and len(conversation_id) > 12 else (conversation_id or "n/a")
        total_tool_calls = (len(tool_calls) if tool_calls else 0) + state.tool_call_count
        print(f"[DEBUG] ===== STREAM SUMMARY (request_id: {rid}, conversation_id: {cid}) =====")
        print(f"[DEBUG]   Chunks: {state.chunk_count}")
        print(f"[DEBUG]   Content: {len(state.accumulated_content)} chars")
        print(f"[DEBUG]   Tool calls: {total_tool_calls} (backend: {state.tool_call_count}, parsed: {len(tool_calls) if tool_calls else 0})")
        print(f"[DEBUG]   Finish: {state.finish_reason}")
        print(f"[DEBUG] ==========================")
    
    # Always send [DONE]
    yield "data: [DONE]\n\n"


def stream_with_tool_transform(
    upstream,
    model_name: str,
    request_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream SSE events from backend and transform tool calls.

    This is the main streaming handler that:
    1. Passes through all chunks as-is
    2. Accumulates content for tool call detection
    3. At stream end, extracts and sends tool calls if needed
    """
    state = StreamState(model_name)
    rid = request_id or "n/a"
    cid = (conversation_id[:12] + "..") if conversation_id and len(conversation_id) > 12 else (conversation_id or "n/a")
    _stream_start = time.perf_counter()
    completed_with_done = False  # True if we received [DONE] from backend

    if DEBUG:
        print(f"[DEBUG] ===== STREAMING START (request_id: {rid}, conversation_id: {cid}) =====")

    try:
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
                        completed_with_done = True
                        if DEBUG:
                            print(f"[DEBUG] Stream received [DONE] from backend (request_id: {rid})")
                        yield from process_stream_end(state, request_id=request_id, conversation_id=conversation_id)
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
                                first_tool_chunk = not state.tool_calls_sent
                                state.tool_calls_sent = True
                                n_this = len(delta["tool_calls"])
                                state.tool_call_count += n_this
                                if DEBUG and first_tool_chunk:
                                    print(
                                        f"[DEBUG] tool_calls stream started (request_id: {rid}, conversation_id: {cid}) "
                                        f"| backend sends many small chunks; first delta has {n_this} entry(ies); "
                                        f"total count in STREAM SUMMARY at end"
                                    )

                            if "content" in delta and delta["content"]:
                                state.accumulated_content += delta["content"]

                        # Pass through as-is
                        yield f"{line}\n"

                    except json.JSONDecodeError:
                        yield f"{line}\n"

            # Stream ended without [DONE] (backend closed connection early, or client disconnect)
            if DEBUG:
                print(f"[DEBUG] ⚠ Stream ended without [DONE] (request_id: {rid}, conversation_id: {cid})")
            yield from process_stream_end(state, request_id=request_id, conversation_id=conversation_id)

        except Exception as e:
            duration_ms = round((time.perf_counter() - _stream_start) * 1000)
            print(f"[ERROR] Backend stream error (request_id: {rid}, conversation_id: {cid}, duration_ms: {duration_ms}): {type(e).__name__}: {e}")
            raise

    finally:
        duration_ms = round((time.perf_counter() - _stream_start) * 1000)
        if DEBUG:
            print(
                f"[DEBUG] Stream generator exit (request_id: {rid}, conversation_id: {cid}, "
                f"completed_with_done: {completed_with_done}, duration_ms: {duration_ms}, chunks: {state.chunk_count})"
            )
        upstream.close()
