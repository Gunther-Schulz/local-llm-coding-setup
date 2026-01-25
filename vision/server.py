"""Vision API Server - FastAPI wrapper around llama.cpp for vision queries."""
import os
import sys
import subprocess
import tempfile
import base64
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Union, Dict, Any

# Import central configuration
from stack.settings import (
    VISION_GGUF_PATH,
    VISION_MMPROJ_PATH,
    VISION_MAX_CONTEXT,
    LLAMACPP_BIN,
    DEBUG
)

# Allow environment override (set by vision launcher)
VISION_MODEL_PATH = os.getenv("VISION_GGUF_PATH", VISION_GGUF_PATH)
VISION_MMPROJ_PATH = os.getenv("VISION_MMPROJ_PATH", VISION_MMPROJ_PATH)
VISION_MAX_CONTEXT = int(os.getenv("VISION_MAX_CONTEXT", str(VISION_MAX_CONTEXT)))
LLAMACPP_BIN = os.getenv("LLAMACPP_BIN", LLAMACPP_BIN)

app = FastAPI(title="Vision API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class VisionMessage(BaseModel):
    """Message with text or multimodal content."""
    role: str
    content: Union[str, List[Dict[str, Any]]]


class VisionRequest(BaseModel):
    """OpenAI-compatible vision request."""
    messages: List[VisionMessage]
    max_tokens: Optional[int] = 512
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9


def extract_image_and_text(messages: List[VisionMessage]) -> tuple:
    """Extract image data and text prompt from messages."""
    text_parts = []
    image_data = None
    
    for msg in messages:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for item in msg.content:
                if isinstance(item, dict):
                    if item.get("type") in ("text", "input_text"):
                        text_parts.append(item.get("text", ""))
                    elif item.get("type") == "image_url":
                        # Extract base64 image data
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict):
                            url = image_url.get("url", "")
                        else:
                            url = image_url
                        
                        if url.startswith("data:image"):
                            # data:image/png;base64,iVBORw0KG...
                            image_data = url.split(",", 1)[1] if "," in url else url
                        else:
                            image_data = url
    
    prompt = " ".join(text_parts).strip()
    if not prompt:
        prompt = "Describe what you see in this image in detail."
    
    return image_data, prompt


def run_vision_query(image_path: str, prompt: str, max_tokens: int = 512) -> str:
    """Run llama.cpp vision query."""
    
    if not os.path.exists(VISION_MODEL_PATH):
        raise HTTPException(status_code=500, detail=f"Vision model not found: {VISION_MODEL_PATH}")
    
    if not os.path.exists(VISION_MMPROJ_PATH):
        raise HTTPException(status_code=500, detail=f"MMProj model not found: {VISION_MMPROJ_PATH}")
    
    if not os.path.exists(LLAMACPP_BIN):
        raise HTTPException(status_code=500, detail=f"llama.cpp binary not found: {LLAMACPP_BIN}")
    
    cmd = [
        LLAMACPP_BIN,
        "-m", VISION_MODEL_PATH,
        "--mmproj", VISION_MMPROJ_PATH,
        "-p", prompt,
        "--image", image_path,
        "-ngl", "0",  # CPU only (main model)
        "-c", str(VISION_MAX_CONTEXT),
        "-n", str(max_tokens),
        "--temp", str(0.7),
        "--top-p", str(0.9),
    ]
    
    if DEBUG:
        print(f"[DEBUG] Running vision query: {' '.join(cmd)}", file=sys.stderr)
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120  # 2 minute timeout
        )
        
        if result.returncode != 0:
            raise HTTPException(
                status_code=500,
                detail=f"Vision model error: {result.stderr}"
            )
        
        # Parse output - llama.cpp outputs the response directly
        response = result.stdout.strip()
        
        # Clean up any system prompts or artifacts
        if "\n\n" in response:
            response = response.split("\n\n", 1)[-1]
        
        return response
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Vision query timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision query failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check."""
    return {
        "status": "ok",
        "vision_model": os.path.basename(VISION_MODEL_PATH) if VISION_MODEL_PATH else "not configured",
        "llamacpp_available": os.path.exists(LLAMACPP_BIN)
    }


@app.post("/v1/chat/completions")
async def chat_completions(request: VisionRequest):
    """Process vision query - OpenAI-compatible endpoint."""
    
    if DEBUG:
        print(f"[DEBUG] Vision request: {len(request.messages)} messages", file=sys.stderr)
    
    # Extract image and prompt
    image_data, prompt = extract_image_and_text(request.messages)
    
    if not image_data:
        raise HTTPException(status_code=400, detail="No image found in request")
    
    # Detect image format from data URL for correct file extension
    suffix = ".png"
    for msg in request.messages:
        c = msg.content if not isinstance(msg, dict) else msg.get("content")
        if isinstance(c, list):
            for item in (m for m in c if isinstance(m, dict) and m.get("type") == "image_url"):
                u = (item.get("image_url") or {})
                u = u.get("url", u) if isinstance(u, dict) else u
                if isinstance(u, str) and ("image/jpeg" in u.split(";", 1)[0].lower() or "image/jpg" in u.split(";", 1)[0].lower()):
                    suffix = ".jpg"
                    break
    
    # Save image to temporary file
    with tempfile.NamedTemporaryFile(mode='wb', suffix=suffix, delete=False) as tmp_file:
        try:
            # Decode base64 image
            img_bytes = base64.b64decode(image_data)
            tmp_file.write(img_bytes)
            tmp_file.flush()
            image_path = tmp_file.name
            
            if DEBUG:
                print(f"[DEBUG] Image saved to: {image_path}", file=sys.stderr)
                print(f"[DEBUG] Prompt: {prompt}", file=sys.stderr)
            
            # Run vision query
            response_text = run_vision_query(
                image_path,
                prompt,
                max_tokens=request.max_tokens
            )
            
            if DEBUG:
                print(f"[DEBUG] Vision response: {response_text[:200]}...", file=sys.stderr)
            
            # Return OpenAI-compatible format
            return {
                "id": "vision-" + os.urandom(12).hex(),
                "object": "chat.completion",
                "created": int(__import__("time").time()),
                "model": "vision",
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_text
                    },
                    "finish_reason": "stop"
                }],
                "usage": {
                    "prompt_tokens": len(prompt.split()),
                    "completion_tokens": len(response_text.split()),
                    "total_tokens": len(prompt.split()) + len(response_text.split())
                }
            }
            
        finally:
            # Clean up temp file
            try:
                os.unlink(image_path)
            except:
                pass
