"""Pydantic models for proxy requests/responses."""
from typing import List, Dict, Any, Optional, Union, Literal
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Chat message with text or multimodal content; supports tool calls and tool results."""
    role: str
    content: Optional[Union[str, List[Dict[str, Any]]]] = None
    name: Optional[str] = None
    # Assistant message with tool use
    tool_calls: Optional[List[Dict[str, Any]]] = None
    # Tool result message (role='tool')
    tool_call_id: Optional[str] = None


class ToolCall(BaseModel):
    """Tool call information."""
    id: str
    type: Literal["function"] = "function"
    function: Dict[str, Any]


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    response_format: Optional[Dict[str, Any]] = None
    n: Optional[int] = 1
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    user: Optional[str] = None
