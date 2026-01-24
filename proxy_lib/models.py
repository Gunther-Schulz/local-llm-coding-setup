"""Pydantic models for API requests/responses"""
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union


class ContentPart(BaseModel):
    """Multimodal content part"""
    type: str
    text: Optional[str] = None
    image_url: Optional[Union[str, Dict[str, Any]]] = None


class Message(BaseModel):
    role: str
    content: Union[str, List[Union[ContentPart, Dict[str, Any]]]]
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None
    
    def get_text_content(self) -> str:
        """Extract text from string or multimodal content"""
        if isinstance(self.content, str):
            return self.content
        
        text_parts = []
        for item in self.content:
            if isinstance(item, dict):
                if item.get("type") in ("text", "input_text") and "text" in item:
                    text_parts.append(item["text"])
            elif isinstance(item, ContentPart):
                if item.type in ("text", "input_text") and item.text:
                    text_parts.append(item.text)
        
        return "\n".join(text_parts)


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    max_tokens: Optional[int] = None
    max_completion_tokens: Optional[int] = None
    stream: Optional[bool] = False
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    tools: Optional[List[Dict]] = None
    tool_choice: Optional[Union[str, Dict]] = None
