"""Interface definitions for proxy services."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class ContextManagerInterface(ABC):
    """Interface for context management: Cursor-style compression on overflow + tool condense."""
    
    @abstractmethod
    def condense_tool_responses_with_context(self, messages: List[Dict], 
                                                 no_condense_patterns: Optional[List[str]] = None) -> List[Dict]:
        """Condense large tool responses while maintaining context."""
        pass

    @abstractmethod
    async def compress_cursor_style(
        self, messages: List[Dict], conversation_id: str, recent_count: int = 6, model: Optional[str] = None
    ) -> List[Dict]:
        """Cursor-style compression: structured summary + last N messages. Used when prompt would exceed backend limit. model: use for summarization (e.g. request.model) or fallback from settings."""
        pass


class VisionRouterInterface(ABC):
    """Interface for vision routing services."""
    
    @abstractmethod
    async def query_vision_api(self, messages: List[Dict], max_tokens: int = 512) -> Dict[str, Any]:
        """Query the vision API for image analysis."""
        pass
    
    @abstractmethod
    def has_image_content(self, messages: List[Dict]) -> bool:
        """Check if any message contains image content."""
        pass
    
    @abstractmethod
    def extract_images_and_text(self, messages: List[Dict]) -> tuple[bool, List[Dict], List[Dict]]:
        """Extract images and text from messages."""
        pass


class ToolParserInterface(ABC):
    """Interface for tool parsing services."""
    
    @abstractmethod
    def should_transform_tool_calls(self, backend_sent_tool_calls: bool) -> bool:
        """Check if tool calls should be transformed."""
        pass
    
    @abstractmethod
    def transform_qwen_response(self, response_data: Dict) -> Dict:
        """Transform Qwen response to extract tool calls."""
        pass


class StreamingHandlerInterface(ABC):
    """Interface for streaming response handling."""
    
    @abstractmethod
    def stream_with_tool_transform(self, upstream, model_name: str, 
                                  request_id: Optional[str] = None, 
                                  conversation_id: Optional[str] = None):
        """Handle streaming responses with tool call transformation."""
        pass