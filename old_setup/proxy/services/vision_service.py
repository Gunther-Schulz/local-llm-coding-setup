"""Vision processing service implementation."""

from typing import List, Dict, Any, Optional
import hashlib
import httpx

from stack.settings import VISION_URL as VISION_API_URL, DEBUG
from proxy.vision_router import (
    _extract_image_hash,
    _hash_image
)


class VisionService:
    """Vision processing service for handling image analysis requests."""
    
    def __init__(self):
        """Initialize vision service."""
        # Cache for vision analysis results (image_hash -> description)
        self._vision_cache: Dict[str, str] = {}
    
    async def query_vision_api(self, messages: List[Dict], max_tokens: int = 512) -> Dict[str, Any]:
        """
        Query the vision API server with caching.
        
        Caches results based on image content hash to avoid re-analyzing the same image.
        """
        # Check cache first
        image_hash = _extract_image_hash(messages)
        if image_hash and image_hash in self._vision_cache:
            if DEBUG:
                print(f"[DEBUG] Vision cache HIT for image {image_hash}")
            return {
                "choices": [{
                    "message": {
                        "content": self._vision_cache[image_hash]
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
                        self._vision_cache[image_hash] = content
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
    
    def has_image_content(self, messages: List[Dict]) -> bool:
        """Check if any message contains image content."""
        from proxy.vision_router import has_image_content
        return has_image_content(messages)
    
    def extract_images_and_text(self, messages: List[Dict]) -> tuple[bool, List[Dict], List[Dict]]:
        """Extract images and text from messages."""
        from proxy.vision_router import extract_images_and_text
        return extract_images_and_text(messages)