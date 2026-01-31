"""Compression proxy server with vision routing and tool call transformation."""
import json
import traceback
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from proxy.models import ChatCompletionRequest
from stack.settings import (
    DEBUG,
    BACKEND_URL,
    MAX_PROMPT_TOKENS,
    get_effective_context_limit,
    COMPRESSION_THRESHOLD,
    SAFETY_MARGIN
)
from proxy.utils import total_tokens, extract_text_from_content, get_conversation_id
from proxy.context_manager import manage_context
from proxy.vision_router import (
    has_image_content, extract_images_and_text,
    query_vision_api, prepare_multimodal_request
)
from proxy.tool_parser import transform_qwen_response
from proxy.streaming import stream_with_tool_transform


# Create FastAPI app
app = FastAPI(title="LLM Compression Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "backend": BACKEND_URL}


@app.get("/v1/models")
async def list_models():
    """Forward models list from backend."""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/v1/models")
            return response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")


@app.post("/v1/chat/completions")
async def chat_completions_v1(request: Request):
    """Handle chat completions with compression, vision routing, and tool transformation."""
    try:
        # Get raw body
        raw_body = await request.body()
        body_json = json.loads(raw_body)
        
        # Log request
        client_ip = request.client.host if request.client else 'unknown'
        print(f"[INFO] Request from {client_ip} - model: {body_json.get('model', 'N/A')}, "
              f"messages: {len(body_json.get('messages', []))}, tools: {'tools' in body_json}")
        
        if DEBUG:
            print(f"[DEBUG] Full request:\n{json.dumps(body_json, indent=2)}")
        
        # Parse request
        chat_request = ChatCompletionRequest(**body_json)
        result = await handle_chat_completions(chat_request)
        
        if DEBUG and isinstance(result, dict):
            print(f"[DEBUG] Response:\n{json.dumps(result, indent=2)}")
        
        return result
    
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        print(f"[ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


async def handle_chat_completions(request: ChatCompletionRequest):
    """Main handler for chat completions."""
    # Convert to dicts
    incoming_messages = []
    for msg in request.messages:
        msg_dict = msg.model_dump()
        # Remove None values that might cause validation issues
        msg_dict = {k: v for k, v in msg_dict.items() if v is not None}
        incoming_messages.append(msg_dict)
    
    # Check for images and process with vision API if present
    if has_image_content(incoming_messages):
        if DEBUG:
            print("[DEBUG] Image detected, routing to vision API")
        
        has_imgs, text_msgs, image_msgs = extract_images_and_text(incoming_messages)
        
        # Query vision API for image description
        vision_result = await query_vision_api(image_msgs, max_tokens=512)
        
        # Check if vision API returned an error
        if "error" in vision_result:
            error_info = vision_result["error"]
            raise HTTPException(
                status_code=503,
                detail=error_info.get("message", "Vision API unavailable")
            )
        
        # Extract vision response content
        vision_content = ""
        if "choices" in vision_result and len(vision_result["choices"]) > 0:
            vision_content = vision_result["choices"][0].get("message", {}).get("content", "")
        
        if DEBUG:
            print(f"[DEBUG] Vision description: {vision_content[:200]}...")
        
        # Replace images with vision descriptions for LLM
        incoming_messages = prepare_multimodal_request(text_msgs, image_msgs, vision_content)
    else:
        # No images - normalize multimodal text content
        for i, msg in enumerate(incoming_messages):
            if isinstance(msg.get("content"), list):
                text_parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                        text_parts.append(item.get("text", ""))
                incoming_messages[i]["content"] = " ".join(text_parts)
    
    # Calculate token budget (uses MODEL_MAX_CONTEXT or MODEL_EXTENDED_CONTEXT)
    BACKEND_CTX_LIMIT = get_effective_context_limit()
    prompt_tokens = total_tokens(incoming_messages, request.tools)
    
    if DEBUG:
        print(f"[DEBUG] Incoming: {len(incoming_messages)} messages, ~{prompt_tokens} tokens")
    
    # Apply Cursor-style context management if conversation is large
    conversation_id = get_conversation_id(incoming_messages)
    final_messages = incoming_messages
    
    # Trigger at 30 messages OR 20K tokens
    if len(incoming_messages) > 30 or prompt_tokens > COMPRESSION_THRESHOLD:
        if DEBUG:
            print(f"[DEBUG] Applying context management (messages: {len(incoming_messages)}, tokens: {prompt_tokens})")
        
        final_messages = await manage_context(
            incoming_messages,
            conversation_id,
            max_messages=20
        )
        
        prompt_tokens = total_tokens(final_messages, request.tools)
        
        if DEBUG:
            print(f"[DEBUG] After context management: {len(final_messages)} messages, ~{prompt_tokens} tokens")
    
    max_completion_allowed = max(1, BACKEND_CTX_LIMIT - SAFETY_MARGIN - prompt_tokens)
    
    if DEBUG:
        print(f"[DEBUG] Final: {len(final_messages)} messages, {prompt_tokens} prompt tokens, {max_completion_allowed} max completion")
    
    # Determine effective max_tokens - be more conservative
    requested = request.max_tokens or request.max_completion_tokens
    
    # Cap requested tokens to what's actually available
    if requested and requested > max_completion_allowed:
        if DEBUG:
            print(f"[DEBUG] Requested {requested} tokens, but only {max_completion_allowed} available - capping")
        requested = max_completion_allowed
    
    effective_max_tokens = requested if requested else min(max_completion_allowed, 2048)
    
    # Clean messages: ensure only valid fields, remove None values
    cleaned_messages = []
    for msg in final_messages:
        cleaned = {
            "role": msg.get("role"),
            "content": msg.get("content", "")
        }
        # Add optional fields only if present
        if msg.get("tool_calls"):
            cleaned["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            cleaned["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name"):
            cleaned["name"] = msg["name"]
        cleaned_messages.append(cleaned)
    
    # Build backend request
    backend_request = {
        "model": request.model,
        "messages": cleaned_messages,
        "max_tokens": effective_max_tokens,
        "stream": bool(request.stream)
    }
    
    # Optional parameters
    if request.temperature is not None:
        backend_request["temperature"] = request.temperature
    if request.top_p is not None:
        backend_request["top_p"] = request.top_p
    if request.stop is not None:
        backend_request["stop"] = request.stop
    if request.tools is not None:
        backend_request["tools"] = request.tools
    if request.tool_choice is not None:
        backend_request["tool_choice"] = request.tool_choice
    if request.frequency_penalty is not None:
        backend_request["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        backend_request["presence_penalty"] = request.presence_penalty
    if request.response_format is not None:
        backend_request["response_format"] = request.response_format
    
    # Send to backend
    try:
        # Handle streaming separately (don't make two requests!)
        if request.stream:
            import requests
            sync_response = requests.post(
                f"{BACKEND_URL}/v1/chat/completions",
                json=backend_request,
                headers={"Content-Type": "application/json"},
                stream=True,
                timeout=180.0
            )
            sync_response.raise_for_status()
            
            return StreamingResponse(
                stream_with_tool_transform(sync_response, request.model),
                media_type="text/event-stream"
            )
        else:
            # Non-streaming
            async with httpx.AsyncClient(timeout=180.0) as client:
                backend_response = await client.post(
                    f"{BACKEND_URL}/v1/chat/completions",
                    json=backend_request,
                    headers={"Content-Type": "application/json"}
                )
                backend_response.raise_for_status()
                
                # Transform tool calls if needed
                response_data = backend_response.json()
                response_data = transform_qwen_response(response_data)
                return response_data
    
    except httpx.HTTPStatusError as e:
        print(f"[ERROR] Backend HTTP error: {e.response.status_code}")
        print(f"[ERROR] {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Backend error: {e.response.text}"
        )
    except Exception as e:
        print(f"[ERROR] Backend request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")
