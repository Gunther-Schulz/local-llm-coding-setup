"""Streaming response handling service implementation."""

from typing import Generator, Optional

from proxy.streaming import stream_with_tool_transform as _stream_with_tool_transform


class StreamingService:
    """Streaming response handling service."""

    def __init__(self):
        """Initialize streaming service."""
        pass

    def stream_with_tool_transform(
        self,
        upstream,
        model_name: str,
        request_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Stream SSE events from backend and transform tool calls.

        Delegates to proxy.streaming.stream_with_tool_transform.
        """
        yield from _stream_with_tool_transform(
            upstream,
            model_name,
            request_id=request_id,
            conversation_id=conversation_id,
        )
