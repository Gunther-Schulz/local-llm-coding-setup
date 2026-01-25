"""Vision detection and routing logic."""
import hashlib
import httpx
from typing import List, Dict, Any, Tuple

from stack.settings import VISION_URL as VISION_API_URL, DEBUG

# Cache for vision analysis results (image_hash -> description)
_vision_cache: Dict[str, str] = {}


def has_image_content(messages: List[Dict[str, Any]]) -> bool:
    """Check if any message contains image content."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False


def extract_images_and_text(messages: List[Dict[str, Any]]) -> Tuple[bool, List[Dict], List[Dict]]:
    """
    Extract images and text from messages.
    
    Returns:
        (has_images, text_messages, image_messages)
    """
    text_messages = []
    image_messages = []
    has_images = False
    
    for msg in messages:
        content = msg.get("content")
        
        if isinstance(content, str):
            # Plain text message
            text_messages.append(msg)
        elif isinstance(content, list):
            # Multimodal content
            text_parts = []
            has_image_in_msg = False
            
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in ("text", "input_text"):
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        has_images = True
                        has_image_in_msg = True
            
            if has_image_in_msg:
                # Keep original multimodal message for vision API
                image_messages.append(msg)
            
            if text_parts:
                # Create text-only version for context
                text_messages.append({
                    "role": msg.get("role"),
                    "content": " ".join(text_parts)
                })
    
    return has_images, text_messages, image_messages


def _hash_image(image_data: str) -> str:
    """Create a hash of image data for caching."""
    return hashlib.sha256(image_data.encode()).hexdigest()[:16]


def _extract_image_hash(messages: List[Dict[str, Any]]) -> str:
    """Extract and hash the image content from messages for cache key."""
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    image_url = item.get("image_url", {})
                    if isinstance(image_url, dict):
                        url = image_url.get("url", "")
                    else:
                        url = image_url
                    # Hash the image data URL
                    return _hash_image(url)
    return ""


async def query_vision_api(messages: List[Dict[str, Any]], max_tokens: int = 512) -> Dict[str, Any]:
    """
    Query the vision API server with caching.
    
    Caches results based on image content hash to avoid re-analyzing the same image.
    """
    # Check cache first
    image_hash = _extract_image_hash(messages)
    if image_hash and image_hash in _vision_cache:
        if DEBUG:
            print(f"[DEBUG] Vision cache HIT for image {image_hash}")
        return {
            "choices": [{
                "message": {
                    "content": _vision_cache[image_hash]
                }
            }]
        }
    
    if DEBUG and image_hash:
        print(f"[DEBUG] Vision cache MISS for image {image_hash}, analyzing...")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{VISION_API_URL}/v1/chat/completions",
                json={
                    "messages": messages,
                    "max_tokens": max_tokens
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Cache the result
            if image_hash and "choices" in result and result["choices"]:
                content = result["choices"][0].get("message", {}).get("content", "")
                if content:
                    _vision_cache[image_hash] = content
                    if DEBUG:
                        print(f"[DEBUG] Cached vision result for image {image_hash}")
            
            return result
    except httpx.ConnectError:
        if DEBUG:
            print(f"Vision API not available at {VISION_API_URL}")
        return {
            "error": {
                "message": "Vision API not available. Image analysis requires the vision server to be running.",
                "type": "vision_unavailable",
                "code": "vision_server_not_running"
            }
        }
    except Exception as e:
        if DEBUG:
            print(f"Vision API error: {e}")
        return {
            "error": {
                "message": f"Vision API error: {str(e)}",
                "type": "vision_error"
            }
        }


def prepare_multimodal_request(
    text_messages: List[Dict[str, Any]],
    image_messages: List[Dict[str, Any]],
    vision_result: str
) -> List[Dict[str, Any]]:
    """
    Prepare request for text LLM with vision analysis.
    
    Replaces image content in the user's message with the vision model's textual description.
    """
    messages = text_messages.copy()
    
    # Replace the image reference in the last user message with vision description
    if messages and messages[-1]["role"] == "user":
        # Replace or append the vision description
        original_content = messages[-1]["content"]
        
        # If the message is asking about an image, replace it with context
        if original_content.strip():
            messages[-1]["content"] = f"{original_content}\n\n<image_context>\n{vision_result}\n</image_context>"
        else:
            # If no text, just use the vision description
            messages[-1]["content"] = f"<image_context>\n{vision_result}\n</image_context>"
    
    return messages
