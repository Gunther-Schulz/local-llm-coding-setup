"""Vision routing for compression proxy"""
import os
from typing import Dict, List, Any, Optional

import httpx

VISION_API_URL = os.getenv("VISION_API_URL", "http://localhost:8004")
DEBUG_MODE = os.getenv("DEBUG", "0") == "1"

def has_image_content(messages: List[Dict]) -> bool:
    """Check if any message contains image content"""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    return True
    return False

def extract_images_and_text(messages: List[Dict]) -> tuple:
    """
    Extract images and text from messages
    Returns: (has_images, text_messages, image_messages)
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
                # Create text-only version for LLM
                text_messages.append({
                    "role": msg.get("role"),
                    "content": " ".join(text_parts)
                })
    
    return has_images, text_messages, image_messages

async def query_vision_api(messages: List[Dict], max_tokens: int = 512) -> str:
    """Query vision API and get description (async, non-blocking)"""
    
    if DEBUG_MODE:
        print(f"[DEBUG] Querying vision API at {VISION_API_URL}")
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:  # 2 min for CPU inference
            response = await client.post(
                f"{VISION_API_URL}/v1/vision/query",
                json={"messages": messages, "max_tokens": max_tokens},
            )
        
        if response.status_code != 200:
            error_msg = f"Vision API error: {response.status_code}"
            if DEBUG_MODE:
                print(f"[DEBUG] {error_msg}: {response.text}")
            return f"[Vision analysis unavailable: {error_msg}]"
        
        result = response.json()
        vision_text = result.get("response", "")
        
        if DEBUG_MODE:
            print(f"[DEBUG] Vision API response: {vision_text[:200]}...")
        
        return vision_text
        
    except httpx.TimeoutException:
        return "[Vision analysis timed out - image processing on CPU can take 10-30 seconds]"
    except httpx.ConnectError as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Vision API connect error: {str(e)}")
        return f"[Vision analysis unavailable: Vision API not running at {VISION_API_URL}. Start with: ./start-vision-api.sh or ./start-all-with-vision.sh]"
    except Exception as e:
        if DEBUG_MODE:
            print(f"[DEBUG] Vision API error: {str(e)}")
        return f"[Vision analysis unavailable: {str(e)}]"

def prepare_multimodal_request(
    original_messages: List[Dict],
    vision_description: str
) -> List[Dict]:
    """
    Prepare messages for LLM by replacing images with vision descriptions
    """
    processed_messages = []
    
    for msg in original_messages:
        content = msg.get("content")
        
        if isinstance(content, str):
            # Plain text - pass through
            processed_messages.append(msg)
        elif isinstance(content, list):
            # Multimodal - extract text and add vision description
            text_parts = []
            had_image = False
            
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") in ("text", "input_text"):
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        had_image = True
            
            # Combine text with vision description
            if had_image:
                combined_text = f"[Image Description: {vision_description}]"
                if text_parts:
                    combined_text = " ".join(text_parts) + "\n\n" + combined_text
            else:
                combined_text = " ".join(text_parts)
            
            processed_messages.append({
                "role": msg.get("role"),
                "content": combined_text
            })
    
    return processed_messages
