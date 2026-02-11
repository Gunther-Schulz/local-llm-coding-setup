"""Compression proxy server with vision routing. Backend (llama-server with --jinja) returns native tool_calls."""
import json
import time
import traceback
import uuid
import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Dict, Any, Optional, List

from proxy.models import ChatCompletionRequest
from stack.settings import (
    DEBUG,
    BACKEND_URL,
    MAX_PROMPT_TOKENS,
    get_effective_context_limit,
    COMPRESSION_ENABLED,
    CONTEXT_WINDOW_SIZE,
    SAFETY_MARGIN,
    INJECT_SYSTEM_MESSAGE,
    SYSTEM_MESSAGE_TEXT,
    INJECT_CAPABILITY_REMINDER,
    CAPABILITY_REMINDER_TEXT,
    VIRTUAL_TOOL_ENABLED,
)
from stack.models import get_model_proxy_flags
from proxy.utils import total_tokens, extract_text_from_content, get_conversation_id
from proxy.context_manager import (
    get_stored,
    execute_virtual_tool,
    VIRTUAL_TOOL_NAME,
    VIRTUAL_TOOL_DEFINITION,
)
from proxy.services.context_service import ContextService
from proxy.services.vision_service import VisionService
from proxy.services.streaming_service import StreamingService
from proxy.services.interfaces import (
    ContextManagerInterface,
    VisionRouterInterface,
    StreamingHandlerInterface
)


def _inject_virtual_tool_results(messages: List[Dict], conversation_id: str) -> tuple[List[Dict], bool]:
    """
    If the last message is assistant with tool_calls and one of them is our virtual tool
    without a result, execute it and inject the tool result. Returns (modified_messages, injected_any).
    """
    if not messages:
        return messages, False
    idx = len(messages) - 1
    while idx >= 0 and (messages[idx].get("role") != "assistant" or not messages[idx].get("tool_calls")):
        idx -= 1
    if idx < 0:
        return messages, False
    assistant_msg = messages[idx]
    following = messages[idx + 1:]
    tool_results_ordered = []
    used = set()
    injected = False
    for tc in assistant_msg["tool_calls"]:
        tid = tc.get("id")
        fn = tc.get("function")
        name = (fn.get("name") or "") if isinstance(fn, dict) else ""
        # Find matching tool message in following
        match_msg = None
        for j, m in enumerate(following):
            if j in used:
                continue
            if m.get("role") == "tool" and m.get("tool_call_id") == tid:
                match_msg = m
                used.add(j)
                break
        if match_msg is not None:
            tool_results_ordered.append(match_msg)
        elif name == VIRTUAL_TOOL_NAME:
            args = {}
            if isinstance(fn, dict) and fn.get("arguments"):
                try:
                    args = json.loads(fn["arguments"]) if isinstance(fn["arguments"], str) else fn["arguments"]
                except (json.JSONDecodeError, TypeError):
                    pass
            content = execute_virtual_tool(conversation_id, args)
            tool_results_ordered.append({"role": "tool", "tool_call_id": tid, "content": content, "name": VIRTUAL_TOOL_NAME})
            injected = True
        # else: missing result for non-virtual tool; don't inject
    if not injected:
        return messages, False
    new_messages = messages[: idx + 1] + tool_results_ordered + [m for j, m in enumerate(following) if j not in used]
    return new_messages, True


# Create FastAPI app
app = FastAPI(title="LLM Compression Proxy", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Import service manager for dependency injection
from proxy.services import service_manager

# Initialize services via service manager
context_service = service_manager.context_service
vision_service = service_manager.vision_service
streaming_service = service_manager.streaming_service


@app.on_event("startup")
async def startup_log():
    """Log proxy config when DEBUG so logs show what is active (global defaults; per-model overrides in models.conf)."""
    if DEBUG:
        ctx = get_effective_context_limit()
        comp = "cursor_style_on_overflow" if COMPRESSION_ENABLED else "off"
        vt = "on" if VIRTUAL_TOOL_ENABLED else "off"
        print(f"[DEBUG] Proxy config: backend={BACKEND_URL}, context_limit={ctx}, compression={comp}, virtual_tool={vt}")
        if INJECT_SYSTEM_MESSAGE:
            if SYSTEM_MESSAGE_TEXT:
                print(f"[DEBUG] System message injection: on (from config)")
            else:
                print(f"[DEBUG] System message injection: on but SYSTEM_MESSAGE_FILE empty or missing – not injecting")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "backend": BACKEND_URL}


@app.get("/v1/models")
async def list_models(request: Request):
    """Forward models list from backend as-is."""
    client_ip = request.client.host if request.client else "unknown"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{BACKEND_URL}/v1/models")
            data = response.json()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")
    if "data" in data and isinstance(data["data"], list):
        print(f"[INFO] GET /v1/models from {client_ip}: {len(data['data'])} model(s)")
    return data


@app.get("/models")
async def list_models_no_v1(request: Request):
    """Alias for clients that request /models instead of /v1/models (e.g. some Cursor configs)."""
    return await list_models(request)


@app.post("/v1/chat/completions")
async def chat_completions_v1(request: Request):
    """Handle chat completions with compression, vision routing, and tool transformation."""
    request_id = request.headers.get("x-request-id") or f"req-{uuid.uuid4().hex[:12]}"
    try:
        # Get raw body
        raw_body = await request.body()
        body_json = json.loads(raw_body)
        
        # Log request (include request_id for correlation with Cursor / backend logs)
        client_ip = request.client.host if request.client else 'unknown'
        print(f"[INFO] Request {request_id} from {client_ip} - model: {body_json.get('model', 'N/A')}, "
              f"messages: {len(body_json.get('messages', []))}, tools: {'tools' in body_json}")
        
        if DEBUG:
            # Log request headers (to discover Cursor vs other clients, e.g. for system-message injection)
            headers_dict = dict(request.headers) if request.headers else {}
            print(f"[DEBUG] Request headers:\n{json.dumps(headers_dict, indent=2)}")
            print(f"[DEBUG] Full request:\n{json.dumps(body_json, indent=2)}")
        
        # Parse request
        chat_request = ChatCompletionRequest(**body_json)
        result = await handle_chat_completions(chat_request, request_id=request_id)
        
        if DEBUG and isinstance(result, dict):
            print(f"[DEBUG] Response:\n{json.dumps(result, indent=2)}")
        
        return result
    
    except HTTPException:
        raise  # Preserve 413, 502, 503 etc. – do not turn into 500
    except Exception as e:
        print(f"[ERROR] Request {request_id} failed: {e}")
        print(f"[ERROR] {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat/completions")
async def chat_completions_no_v1(request: Request):
    """Alias for clients that request /chat/completions instead of /v1/chat/completions (e.g. some Cursor configs)."""
    return await chat_completions_v1(request)


async def handle_chat_completions(request: ChatCompletionRequest, request_id: Optional[str] = None):
    """Main handler for chat completions."""
    _t0 = time.perf_counter()
    # Per-model proxy flags (from config/models.conf optional columns); None = use global
    model_flags = get_model_proxy_flags(request.model or "")
    effective_compression = model_flags.get("compression") if model_flags.get("compression") is not None else COMPRESSION_ENABLED
    effective_virtual_tool = (model_flags.get("virtual_tool") if model_flags.get("virtual_tool") is not None else VIRTUAL_TOOL_ENABLED) and VIRTUAL_TOOL_ENABLED
    effective_inject_system = model_flags.get("inject_system") if model_flags.get("inject_system") is not None else INJECT_SYSTEM_MESSAGE
    effective_inject_capability = model_flags.get("inject_capability") if model_flags.get("inject_capability") is not None else INJECT_CAPABILITY_REMINDER

    # Convert to dicts, preserving tool_calls and tool_call_id for multi-turn tool use
    incoming_messages = []
    for msg in request.messages:
        msg_dict = msg.model_dump()
        # Keep all fields; only drop explicit None for optional content (use "" for backend)
        if msg_dict.get("content") is None:
            msg_dict["content"] = ""
        msg_dict = {k: v for k, v in msg_dict.items() if v is not None}
        incoming_messages.append(msg_dict)
    if DEBUG and any(m.get("tool_calls") or m.get("tool_call_id") for m in incoming_messages):
        print(f"[DEBUG] Request has tool context: {sum(1 for m in incoming_messages if m.get('tool_calls'))} assistant tool_calls, "
              f"{sum(1 for m in incoming_messages if m.get('tool_call_id'))} tool result(s)")
    
    # Check for images and process with vision API if present
    if vision_service.has_image_content(incoming_messages):
        if DEBUG:
            print("[DEBUG] Image detected, routing to vision API")
        _t_vision = time.perf_counter()

        has_imgs, text_msgs, image_msgs = vision_service.extract_images_and_text(incoming_messages)

        # Query vision API for image description (this call can take 10s–2min; timeout 120s)
        vision_result = await vision_service.query_vision_api(image_msgs, max_tokens=512)
        if DEBUG:
            print(f"[DEBUG] Vision API took {round((time.perf_counter() - _t_vision) * 1000)} ms")
        
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
        # We'll use the original function since it's not in the service
        from proxy.vision_router import prepare_multimodal_request
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
    
    # Cursor-style compression: when effective_compression, condense tool responses then compress on overflow only.
    tokens_before_condense = total_tokens(incoming_messages, request.tools)
    BACKEND_CTX_LIMIT = get_effective_context_limit()
    if effective_compression:
        incoming_messages = context_service.condense_tool_responses_with_context(incoming_messages)
    tool_responses_condensed = sum(1 for m in incoming_messages if m.get("_full_content_length") is not None)
    tokens_after_condense = total_tokens(incoming_messages, request.tools)
    prompt_tokens = tokens_after_condense

    conversation_id = get_conversation_id(incoming_messages)
    final_messages = incoming_messages
    cursor_style_compressed = False

    max_completion_allowed = max(1, BACKEND_CTX_LIMIT - SAFETY_MARGIN - prompt_tokens)
    requested = request.max_tokens or request.max_completion_tokens
    if requested and requested > max_completion_allowed:
        if DEBUG:
            print(f"[DEBUG] Requested {requested} tokens, but only {max_completion_allowed} available - capping")
        requested = max_completion_allowed
    effective_max_tokens = requested if requested else max_completion_allowed
    MIN_MAX_TOKENS = 64
    if effective_max_tokens < MIN_MAX_TOKENS and max_completion_allowed < MIN_MAX_TOKENS:
        # Would return 413: try Cursor-style compression if enabled
        if effective_compression:
            if DEBUG:
                print(f"[DEBUG] Prompt over limit ({prompt_tokens}), applying Cursor-style compression")
            final_messages = await context_service.compress_cursor_style(
                incoming_messages, conversation_id, recent_count=CONTEXT_WINDOW_SIZE, model=request.model
            )
            prompt_tokens = total_tokens(final_messages, request.tools)
            max_completion_allowed = max(1, BACKEND_CTX_LIMIT - SAFETY_MARGIN - prompt_tokens)
            cursor_style_compressed = True
            if max_completion_allowed < MIN_MAX_TOKENS:
                if DEBUG:
                    print(f"[DEBUG] Rejecting: still too long after compression ({prompt_tokens} > {BACKEND_CTX_LIMIT})")
                raise HTTPException(
                    status_code=413,
                    detail=f"Prompt too long: {prompt_tokens} tokens exceeds backend context ({BACKEND_CTX_LIMIT}). "
                    "Shorten the conversation or use fewer tool results."
                )
            effective_max_tokens = min(MIN_MAX_TOKENS, max_completion_allowed)
        else:
            if DEBUG:
                print(f"[DEBUG] Rejecting: prompt too long ({prompt_tokens} > {BACKEND_CTX_LIMIT})")
            raise HTTPException(
                status_code=413,
                detail=f"Prompt too long: {prompt_tokens} tokens exceeds backend context ({BACKEND_CTX_LIMIT}). "
                "Shorten the conversation or use fewer tool results."
            )
    elif effective_max_tokens < MIN_MAX_TOKENS:
        effective_max_tokens = min(MIN_MAX_TOKENS, max_completion_allowed)

    if DEBUG:
        elapsed_ms = round((time.perf_counter() - _t0) * 1000)
        conv_id_short = (conversation_id[:12] + "..") if len(conversation_id) > 12 else conversation_id
        rid = request_id or "n/a"
        print(f"[DEBUG] ---- context (request_id: {rid}) ----")
        print(f"[DEBUG]   stream: {bool(request.stream)}, conversation_id: {conv_id_short}, elapsed_ms: {elapsed_ms}")
        print(f"[DEBUG]   messages_in: {len(incoming_messages)}, tokens_in: {tokens_before_condense}")
        print(f"[DEBUG]   tool_condense: {tool_responses_condensed} response(s) condensed, tokens_after_condense: {tokens_after_condense}")
        if cursor_style_compressed:
            print(f"[DEBUG]   cursor_style_overflow: yes (recent_count: {CONTEXT_WINDOW_SIZE})")
        print(f"[DEBUG]   messages_out: {len(final_messages)}, tokens_out: {prompt_tokens}")
        print(f"[DEBUG]   context_limit: {BACKEND_CTX_LIMIT}, max_completion_available: {max_completion_allowed}, max_tokens_sent: {effective_max_tokens}")
        print("[DEBUG] --------------------------")

    # Virtual tool: if enabled and last message is assistant with tool_calls including search_compressed_conversation
    # and no result for it, execute and inject the tool result so the backend sees it.
    final_messages, virtual_injected = _inject_virtual_tool_results(final_messages, conversation_id) if effective_virtual_tool else (final_messages, False)
    if virtual_injected and DEBUG:
        print(f"[DEBUG] Injected virtual tool result for conversation_id={conversation_id[:12]}...")

    # Clean messages: ensure only valid fields; match backend expectations for tool turns
    cleaned_messages = []
    for msg in final_messages:
        role = msg.get("role")
        content = msg.get("content", "")
        cleaned = {"role": role}
        # Assistant with only tool_calls: omit content so backend sees tool turn (llama-server can mis-handle content: "")
        if role == "assistant" and msg.get("tool_calls") and (content is None or content == ""):
            pass  # no content key
        else:
            cleaned["content"] = content if content is not None else ""
        if msg.get("tool_calls"):
            cleaned["tool_calls"] = msg["tool_calls"]
        if msg.get("tool_call_id"):
            cleaned["tool_call_id"] = msg["tool_call_id"]
        if msg.get("name"):
            cleaned["name"] = msg["name"]
        cleaned_messages.append(cleaned)

    # Optional: inject Cursor-style system message when using clients that don't send one (e.g. Continue).
    if effective_inject_system and SYSTEM_MESSAGE_TEXT:
        if cleaned_messages and cleaned_messages[0].get("role") == "system":
            cleaned_messages[0]["content"] = SYSTEM_MESSAGE_TEXT
        else:
            cleaned_messages.insert(0, {"role": "system", "content": SYSTEM_MESSAGE_TEXT})

    # Optional: remind model it has conversation context and tools (reduces "I can't recall" / "I can't search" / "I'll just advise")
    if (
        effective_inject_capability
        and (request.tools or (effective_virtual_tool and get_stored(conversation_id)))
        and cleaned_messages
        and cleaned_messages[0].get("role") == "system"
    ):
        existing = cleaned_messages[0].get("content") or ""
        if isinstance(existing, str) and CAPABILITY_REMINDER_TEXT.strip() not in existing:
            cleaned_messages[0]["content"] = existing.rstrip() + CAPABILITY_REMINDER_TEXT

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
        tools_list = list(request.tools)
        # When virtual tool enabled and we have stored compressed conversation, add virtual tool so the model can query it
        if effective_virtual_tool and get_stored(conversation_id) and not any(
            t.get("function", {}).get("name") == VIRTUAL_TOOL_NAME
            for t in tools_list
            if isinstance(t, dict) and t.get("type") == "function"
        ):
            tools_list.append(VIRTUAL_TOOL_DEFINITION)
        backend_request["tools"] = tools_list
    elif effective_virtual_tool and get_stored(conversation_id):
        backend_request["tools"] = [VIRTUAL_TOOL_DEFINITION]
        backend_request["tool_choice"] = "auto"
    if request.tool_choice is not None and "tool_choice" not in backend_request:
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
            if not sync_response.ok:
                body = sync_response.text
                rid = request_id or "n/a"
                print(f"[ERROR] Request {rid} Backend HTTP error (streaming): {sync_response.status_code}")
                print(f"[ERROR] Backend response: {body[:2000]}")
                raise HTTPException(
                    status_code=502,
                    detail=f"Backend error ({sync_response.status_code}): {body[:500]}"
                )
            if DEBUG:
                elapsed_ms = round((time.perf_counter() - _t0) * 1000)
                conv_short = (conversation_id[:12] + "..") if conversation_id and len(conversation_id) > 12 else (conversation_id or "n/a")
                print(
                    f"[DEBUG] Backend 200, forwarding stream (request_id: {request_id or 'n/a'}, "
                    f"conversation_id: {conv_short}, elapsed_ms: {elapsed_ms})"
                )
            return StreamingResponse(
                service_manager.streaming_service.stream_with_tool_transform(
                    sync_response, request.model,
                    request_id=request_id, conversation_id=conversation_id
                ),
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
                response_data = backend_response.json()
                if DEBUG:
                    print(f"[DEBUG] Request {request_id or 'n/a'} completed in {round((time.perf_counter() - _t0) * 1000)} ms (non-streaming)")
                return response_data
    
    except httpx.HTTPStatusError as e:
        rid = request_id or "n/a"
        print(f"[ERROR] Request {rid} Backend HTTP error (non-streaming): {e.response.status_code}")
        print(f"[ERROR] {e.response.text}")
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Backend error: {e.response.text}"
        )
    except Exception as e:
        rid = request_id or "n/a"
        print(f"[ERROR] Request {rid} Backend request failed: {e}")
        raise HTTPException(status_code=502, detail=f"Backend error: {str(e)}")
