"""Proxy services package."""

from .context_service import ContextService
from .vision_service import VisionService
from .tool_service import ToolService
from .streaming_service import StreamingService

# Service manager to provide single point of access
class ServiceManager:
    """Manages proxy services."""
    
    def __init__(self):
        self._context_service = None
        self._vision_service = None
        self._tool_service = None
        self._streaming_service = None
    
    @property
    def context_service(self):
        if self._context_service is None:
            self._context_service = ContextService()
        return self._context_service
    
    @property
    def vision_service(self):
        if self._vision_service is None:
            self._vision_service = VisionService()
        return self._vision_service
    
    @property
    def tool_service(self):
        if self._tool_service is None:
            self._tool_service = ToolService()
        return self._tool_service
    
    @property
    def streaming_service(self):
        if self._streaming_service is None:
            self._streaming_service = StreamingService()
        return self._streaming_service

# Global service manager instance
service_manager = ServiceManager()