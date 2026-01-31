"""Interface definitions for proxy services."""

from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class ContextManagerInterface(ABC):
    """Interface for context management services."""
    
    @abstractmethod
    async def manage_context(self, messages: List[Dict], conversation_id: str, 
                           max_messages: int = 20) -> List[Dict]:
        """Manage conversation context with sliding window and summarization."""
        pass
    
    @abstractmethod
    async def condense_tool_responses_with_context(self, messages: List[Dict], 
                                                 no_condense_patterns: Optional[List[str]] = None) -> List[Dict]:
        """Condense large tool responses while maintaining context."""
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
    def should_transform_tool_calls(self, vllm_sent: bool) -> bool:
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