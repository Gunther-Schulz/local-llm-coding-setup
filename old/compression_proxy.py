"""
Compression Proxy for vLLM
Provides OpenAI API compatibility with context compression and tool call transformation
"""
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import requests
import json
import traceback
import argparse
import sys

# Import proxy modules
from proxy_lib.config import (
    DEBUG_MODE, BACKEND_SERVER_URL, MAX_PROMPT_TOKENS, MODEL_MAX_CONTEXT,
    COMPRESSION_THRESHOLD, MODEL_TOOL_FORMAT
)
from proxy_lib.models import ChatCompletionRequest
from proxy_lib.utils import get_conversation_id, total_tokens
from proxy_lib.compression import manage_conversation_history
from proxy_lib.streaming import stream_with_tool_transform
from proxy_lib.tool_parser import transform_qwen_response
from proxy_lib.vision_router import (
    has_image_content, extract_images_and_text,
    query_vision_api, prepare_multimodal_request
)

# FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "ok"}


@app.post("/v1/chat/completions")
async def chat_completions_v1(request: Request):
    """Handle chat completions with logging and error handling"""
    try:
        # Get raw body for logging
        raw_body = await request.body()
        body_json = json.loads(raw_body)
        
        # Log request
        client_ip = request.client.host if request.client else 'unknown'
        print(f"[INFO] Request from {client_ip} - model: {body_json.get('model', 'N/A')}, "
              f"messages: {len(body_json.get('messages', []))}, tools: {'tools' in body_json}")
        
        if DEBUG_MODE:
            print(f"[DEBUG] Full request:\n{json.dumps(body_json, indent=2)}")
        
        # Parse and handle
        chat_request = ChatCompletionRequest(**body_json)
        result = await handle_chat_completions(chat_request)
        
        if DEBUG_MODE and isinstance(result, dict):
            print(f"[DEBUG] Response:\n{json.dumps(result, indent=2)}")
        
        return result
    
    except Exception as e:
        print(f"[ERROR] Request failed: {e}")
        print(f"[ERROR] {traceback.format_exc()}")
        raise


async def handle_chat_completions(request: ChatCompletionRequest):
    """Main handler for chat completions"""
    try:
        # Convert to dicts
        incoming_messages = []
        for msg in request.messages:
            msg_dict = msg.model_dump()
            # Remove None values that might cause validation issues
            msg_dict = {k: v for k, v in msg_dict.items() if v is not None}
            incoming_messages.append(msg_dict)
        
        # Check for images and process with vision API if present
        if has_image_content(incoming_messages):
            if DEBUG_MODE:
                print("[DEBUG] Image detected, routing to vision API")
            
            has_imgs, text_msgs, image_msgs = extract_images_and_text(incoming_messages)
            
            # Query vision API for image description
            vision_description = await query_vision_api(image_msgs, max_tokens=512)
            
            if DEBUG_MODE:
                print(f"[DEBUG] Vision description: {vision_description[:200]}...")
            
            # Replace images with vision descriptions for LLM
            incoming_messages = prepare_multimodal_request(incoming_messages, vision_description)
        else:
            # No images - normalize multimodal text content
            for i, msg in enumerate(incoming_messages):
                if isinstance(msg.get("content"), list):
                    text_parts = []
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") in ("text", "input_text"):
                            text_parts.append(item.get("text", ""))
                    incoming_messages[i]["content"] = " ".join(text_parts)
        
        # Get conversation ID and manage history
        conversation_id = get_conversation_id(incoming_messages)
        final_messages = manage_conversation_history(conversation_id, incoming_messages)
        
        # Calculate token budget
        BACKEND_CTX_LIMIT = MODEL_MAX_CONTEXT
        SAFETY_MARGIN = 2048
        prompt_tokens = total_tokens(final_messages, request.tools)
        max_completion_allowed = max(1, BACKEND_CTX_LIMIT - SAFETY_MARGIN - prompt_tokens)
        
        if DEBUG_MODE:
            print(f"[DEBUG] Tokens: prompt={prompt_tokens}, max_completion={max_completion_allowed}")
        
        # If prompt is too large, force compression before rejecting
        if prompt_tokens > MAX_PROMPT_TOKENS:
            if DEBUG_MODE:
                print(f"[DEBUG] Prompt too large ({prompt_tokens}), forcing compression")
            
            from proxy_lib.compression import compress_messages
            final_messages = compress_messages(final_messages, keep_recent=2)
            
            # Recalculate after compression
            prompt_tokens = total_tokens(final_messages, request.tools)
            max_completion_allowed = max(1, BACKEND_CTX_LIMIT - SAFETY_MARGIN - prompt_tokens)
            
            if DEBUG_MODE:
                print(f"[DEBUG] After compression: prompt={prompt_tokens}, max_completion={max_completion_allowed}")
            
            # Still too large? Reject
            if prompt_tokens > MAX_PROMPT_TOKENS:
                raise HTTPException(
                    status_code=400,
                    detail=f"Prompt too large even after compression: {prompt_tokens} tokens (max: {MAX_PROMPT_TOKENS})"
                )
        
        # Determine effective max_tokens - be more conservative
        requested = request.max_tokens or request.max_completion_tokens
        
        # Cap requested tokens to what's actually available
        if requested and requested > max_completion_allowed:
            if DEBUG_MODE:
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
        
        if DEBUG_MODE:
            print(f"[DEBUG] Request to vLLM:")
            print(f"[DEBUG]   Messages: {len(backend_request['messages'])}")
            print(f"[DEBUG]   Max tokens: {backend_request['max_tokens']}")
            print(f"[DEBUG]   Estimated prompt tokens: {prompt_tokens}")
            print(f"[DEBUG]   Total with completion: {prompt_tokens + backend_request['max_tokens']}")
            print(f"[DEBUG]   Context limit: {BACKEND_CTX_LIMIT}")
        
        # Handle streaming
        if request.stream:
            upstream = requests.post(
                f"{BACKEND_SERVER_URL}/v1/chat/completions",
                json=backend_request,
                stream=True,
                timeout=300
            )
            
            if upstream.status_code != 200:
                detail = upstream.text[:500]
                raise HTTPException(status_code=upstream.status_code, detail=detail)
            
            return StreamingResponse(
                stream_with_tool_transform(upstream, request.model),
                media_type="text/event-stream"
            )
        
        # Handle non-streaming
        response = requests.post(
            f"{BACKEND_SERVER_URL}/v1/chat/completions",
            json=backend_request,
            timeout=300
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        
        response_data = response.json()
        
        # Transform tool calls if needed
        if DEBUG_MODE:
            print(f"[DEBUG] Response before transform: {json.dumps(response_data, indent=2)[:500]}")
        
        response_data = transform_qwen_response(response_data)
        
        return response_data
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ERROR] Handler error: {e}")
        print(f"[ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": "default",
            "object": "model",
            "created": 0,
            "owned_by": "local"
        }]
    }


if __name__ == "__main__":
    import uvicorn
    import os
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Compression proxy for vLLM")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    
    # Set debug mode
    if args.debug:
        os.environ["DEBUG"] = "1"
        import proxy_lib.config
        proxy_lib.config.DEBUG_MODE = True
    
    debug_status = os.environ.get("DEBUG", "0") == "1"
    
    # Startup banner - print to stderr so it appears before uvicorn output
    import sys
    print(f"🚀 Starting compression proxy on port 8002", file=sys.stderr)
    print(f"   Backend: {BACKEND_SERVER_URL}", file=sys.stderr)
    print(f"   Model context: {MODEL_MAX_CONTEXT} tokens", file=sys.stderr)
    print(f"   Compression threshold: {COMPRESSION_THRESHOLD} tokens", file=sys.stderr)
    print(f"   Max prompt: {MAX_PROMPT_TOKENS} tokens", file=sys.stderr)
    print(f"   Tool format: {MODEL_TOOL_FORMAT}", file=sys.stderr)
    print(f"   Debug: {'ENABLED' if debug_status else 'DISABLED'}", file=sys.stderr)
    if not debug_status:
        print(f"   (use -d or --debug for full logging)", file=sys.stderr)
    print(file=sys.stderr)
    sys.stderr.flush()
    
    uvicorn.run(app, host="0.0.0.0", port=8002)
