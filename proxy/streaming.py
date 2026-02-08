"""Streaming response handling: pass-through from backend (llama-server returns native tool_calls with --jinja)."""
import json
import time
import uuid
from typing import Generator, Optional

from stack.settings import DEBUG


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
    """Process end of stream: log and send [DONE]. Backend sends native tool_calls."""
    if DEBUG:
        rid = request_id or "n/a"
        cid = (conversation_id[:12] + "..") if conversation_id and len(conversation_id) > 12 else (conversation_id or "n/a")
        print(f"[DEBUG] ===== STREAM SUMMARY (request_id: {rid}, conversation_id: {cid}) =====")
        print(f"[DEBUG]   Chunks: {state.chunk_count}")
        print(f"[DEBUG]   Content: {len(state.accumulated_content)} chars")
        print(f"[DEBUG]   Tool calls: {state.tool_call_count}")
        print(f"[DEBUG]   Finish: {state.finish_reason}")
        print(f"[DEBUG] ==========================")
    yield "data: [DONE]\n\n"


def stream_with_tool_transform(
    upstream,
    model_name: str,
    request_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
) -> Generator[str, None, None]:
    """
    Stream SSE events from backend (pass-through). Backend returns native tool_calls with --jinja.
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
                                state.tool_calls_sent = True
                                state.tool_call_count += len(delta["tool_calls"])

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
