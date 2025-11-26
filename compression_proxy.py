from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Optional, Any, Union
import requests
import hashlib
import json
from llmlingua import PromptCompressor

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

LLAMA_SERVER_URL = "http://localhost:8000"
# Start compressing well before the model's 80K context limit so we never
# send more than ~n_ctx tokens to llama.cpp.
COMPRESSION_THRESHOLD = 40000
# Keep the last few real turns uncompressed; everything older is eligible
# for summarization when we cross the threshold.
KEEP_RECENT_MESSAGES = 4
# Hard safety cap on prompt tokens we send to llama.cpp (n_ctx is 81920)
MAX_PROMPT_TOKENS = 78000

conversation_cache: Dict[str, List[Dict]] = {}
# Rolling summary per conversation (single synthetic message)
conversation_summary: Dict[str, str] = {}
compressed_cache: Dict[str, str] = {}

compressor = PromptCompressor(model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank", use_llmlingua2=True)

class Message(BaseModel):
    role: str
    content: str
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None
    name: Optional[str] = None

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[List[Dict[str, Any]]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = None
    top_p: Optional[float] = None
    frequency_penalty: Optional[float] = None
    presence_penalty: Optional[float] = None

def get_conversation_id(messages: List[Dict]) -> str:
    if len(messages) > 0:
        content = messages[0].get("content", "")[:100]
        return hashlib.md5(content.encode()).hexdigest()
    return "default"

def estimate_tokens(text: str) -> int:
    return len(text) // 4

def compress_messages(messages: List[Dict], keep_recent: int = KEEP_RECENT_MESSAGES) -> List[Dict]:
    """Compress old messages, but preserve tool calls and tool responses"""
    if len(messages) <= keep_recent:
        return messages
    
    recent_messages = messages[-keep_recent:]
    old_messages = messages[:-keep_recent]
    
    print(f"[compression] compress_messages: total={len(messages)}, keep_recent={keep_recent}, to_compress={len(old_messages)}")
    
    compressed_old = []
    for msg in old_messages:
        # Don't compress messages with tool calls or tool responses
        if msg.get("tool_calls") or msg.get("tool_call_id"):
            compressed_old.append(msg)
            continue
        
        msg_id = hashlib.md5(json.dumps(msg, sort_keys=True).encode()).hexdigest()
        
        if msg_id in compressed_cache:
            compressed_content = compressed_cache[msg_id]
        else:
            try:
                compressed = compressor.compress_prompt(msg["content"], rate=0.5, target_token=len(msg["content"].split()) // 2)
                compressed_content = compressed["compressed_prompt"]
                compressed_cache[msg_id] = compressed_content
            except Exception as e:
                print(f"Compression error: {e}")
                compressed_content = msg["content"]
        
        compressed_old.append({"role": msg["role"], "content": f"[Compressed] {compressed_content}"})
    
    return compressed_old + recent_messages

def optimize_context(messages: List[Dict], conversation_id: str) -> List[Dict]:
    if conversation_id not in conversation_cache:
        conversation_cache[conversation_id] = []
    
    conversation_cache[conversation_id].extend(messages)
    
    total_text = " ".join([msg.get("content", "") for msg in conversation_cache[conversation_id]])
    estimated_tokens = estimate_tokens(total_text)
    
    if estimated_tokens > COMPRESSION_THRESHOLD:
        print(f"[compression] optimize_context: convo={conversation_id}, est_tokens={estimated_tokens} > threshold={COMPRESSION_THRESHOLD} -> summarizing history")
        conversation_cache[conversation_id] = compress_messages(conversation_cache[conversation_id])
    
    return conversation_cache[conversation_id]

async def handle_chat_completions(request: ChatCompletionRequest):
    try:
        # Normalize messages into a schema that llama-cpp-python's OpenAI adapter accepts.
        # Some clients (like Cursor) may send richer OpenAI-style messages (tool calls, etc.)
        # that llama-cpp's Pydantic models currently reject. We keep the textual content and
        # roles, but strip problematic fields so the backend always gets valid input.
        raw_messages = []
        for msg in request.messages:
            d = msg.model_dump()
            role = d.get("role")
            content = d.get("content", "")

            # Keep only simple system/user/assistant messages with string content.
            if role in ("system", "user", "assistant"):
                if isinstance(content, str):
                    raw_messages.append({"role": role, "content": content})
                else:
                    # If content is structured (e.g. list segments), fall back to JSON string
                    raw_messages.append({"role": role, "content": json.dumps(content)})
            # Skip tool/function messages entirely to avoid schema conflicts in llama-cpp.

        conversation_id = get_conversation_id(raw_messages)

        # Split into dev prompt (first), middle history, and recent real turns
        dev_prompt = raw_messages[0] if raw_messages else None
        rest = raw_messages[1:] if len(raw_messages) > 1 else []

        recent = rest[-KEEP_RECENT_MESSAGES:] if KEEP_RECENT_MESSAGES > 0 else rest
        middle = rest[:-KEEP_RECENT_MESSAGES] if KEEP_RECENT_MESSAGES > 0 else []

        # Build / update rolling summary from middle + previous summary
        prev_summary = conversation_summary.get(conversation_id, "")
        if middle:
            middle_text = " ".join(m.get("content", "") for m in middle)
            summary_source = (prev_summary + " " + middle_text).strip()
            try:
                compressed = compressor.compress_prompt(
                    summary_source,
                    rate=0.5,
                    target_token=max(1, len(summary_source.split()) // 4),
                )
                new_summary = compressed["compressed_prompt"]
            except Exception as e:
                print(f"Compression error (summary): {e}")
                new_summary = summary_source
            conversation_summary[conversation_id] = new_summary

        summary_text = conversation_summary.get(conversation_id, "").strip()

        optimized_messages: List[Dict] = []
        if dev_prompt is not None:
            optimized_messages.append(dev_prompt)
        if summary_text:
            optimized_messages.append(
                {"role": "system", "content": f"[Conversation summary]\n{summary_text}"}
            )
        optimized_messages.extend(recent)

        # Enforce a hard cap on total prompt tokens before calling llama.cpp
        def total_tokens(msgs: List[Dict]) -> int:
            text = " ".join(m.get("content", "") for m in msgs)
            return estimate_tokens(text)

        # If over MAX_PROMPT_TOKENS, repeatedly compress summary and, if needed,
        # drop the oldest non-dev/summary messages, but never drop dev_prompt.
        current_tokens = total_tokens(optimized_messages)
        if current_tokens > MAX_PROMPT_TOKENS:
            print(f"[compression] hard-cap: initial_tokens={current_tokens}, max={MAX_PROMPT_TOKENS}, messages={len(optimized_messages)}")

        while current_tokens > MAX_PROMPT_TOKENS:
            # Try to compress the summary more if it exists
            if len(optimized_messages) >= 2 and optimized_messages[1]["content"].startswith("[Conversation summary]"):
                try:
                    body = optimized_messages[1]["content"].split("\n", 1)[-1]
                    compressed = compressor.compress_prompt(
                        body,
                        rate=0.5,
                        target_token=max(1, len(body.split()) // 4),
                    )
                    new_summary = compressed["compressed_prompt"]
                    optimized_messages[1]["content"] = f"[Conversation summary]\n{new_summary}"
                except Exception as e:
                    print(f"Compression error (summary hard-cap): {e}")
            # If still too big and we have more than dev + summary + KEEP_RECENT_MESSAGES,
            # drop the oldest non-dev/summary message.
            current_tokens = total_tokens(optimized_messages)
            if current_tokens > MAX_PROMPT_TOKENS:
                if len(optimized_messages) > 2 + KEEP_RECENT_MESSAGES:
                    # Remove the first non-dev/summary message after index 1
                    dropped = optimized_messages.pop(2)
                    print(f"[compression] hard-cap: dropped oldest message (role={dropped.get('role')}) to reduce tokens; now messages={len(optimized_messages)}")
                else:
                    # Nothing left to drop safely; break to avoid infinite loop
                    break

        final_tokens = total_tokens(optimized_messages)
        if final_tokens != current_tokens or final_tokens > MAX_PROMPT_TOKENS:
            print(f"[compression] hard-cap: final_tokens={final_tokens}, messages={len(optimized_messages)}")
        
        # Build request with all parameters, explicitly including tool calling
        # Preserve the client's stream preference
        stream = bool(request.stream)

        llama_request = {
            "model": request.model,
            "messages": optimized_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": stream
        }
        
        # Explicitly pass through tool calling parameters
        if request.tools is not None:
            llama_request["tools"] = request.tools
        
        if request.tool_choice is not None:
            llama_request["tool_choice"] = request.tool_choice
        
        # Pass through other OpenAI-compatible parameters
        if request.top_p is not None:
            llama_request["top_p"] = request.top_p
        
        if request.frequency_penalty is not None:
            llama_request["frequency_penalty"] = request.frequency_penalty
        
        if request.presence_penalty is not None:
            llama_request["presence_penalty"] = request.presence_penalty

        # For streaming requests, transparently proxy SSE without JSON decoding
        if stream:
            upstream = requests.post(
                f"{LLAMA_SERVER_URL}/v1/chat/completions",
                json=llama_request,
                timeout=300,
                stream=True,
            )

            if upstream.status_code != 200:
                # Read text body for error context
                detail = upstream.text
                upstream.close()
                raise HTTPException(status_code=upstream.status_code, detail=detail)

            def iter_events():
                try:
                    for chunk in upstream.iter_content(chunk_size=None):
                        if chunk:
                            yield chunk
                finally:
                    upstream.close()

            return StreamingResponse(iter_events(), media_type="text/event-stream")

        # Non-streaming: expect a single JSON object
        response = requests.post(
            f"{LLAMA_SERVER_URL}/v1/chat/completions",
            json=llama_request,
            timeout=300,
        )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        try:
            return response.json()
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=500,
                detail=f"Invalid JSON response from llama server: {response.text[:500]}",
            )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def handle_list_models():
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/v1/models", timeout=30)
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)
        try:
            return response.json()
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=500, detail=f"Invalid JSON response from llama server: {response.text[:500]}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {"status": "ok", "service": "compression-proxy"}

# Health check
@app.get("/health")
async def health():
    return {"status": "healthy"}

# V1 root
@app.get("/v1")
async def v1_root():
    return {"status": "ok", "service": "compression-proxy", "version": "v1"}

# OpenAI-compatible endpoints (with /v1/ prefix)
@app.post("/v1/chat/completions")
async def chat_completions_v1(request: ChatCompletionRequest):
    return await handle_chat_completions(request)

@app.get("/v1/models")
async def list_models_v1():
    return await handle_list_models()

# OpenAI-compatible endpoints (without /v1/ prefix - for compatibility)
@app.post("/chat/completions")
async def chat_completions_no_v1(request: ChatCompletionRequest):
    return await handle_chat_completions(request)

@app.get("/chat/completions")
async def chat_completions_get():
    raise HTTPException(status_code=405, detail="Method not allowed. Use POST for chat completions.")

@app.get("/models")
async def list_models_no_v1():
    return await handle_list_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)

