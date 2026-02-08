"""models.conf: registry and export for llama-server/proxy. Single source for model definitions."""
from pathlib import Path
from .paths import root

MODELS_CONF = "config/models.conf"
# Format: model_key|display_name|gguf_path|tokenizer_id|max_context|tool_parser|tool_format|download_url|extended_context|description

def _path() -> Path:
    return root() / MODELS_CONF

def _parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|")
    if len(parts) < 10:
        return None
    return {
        "model_key": parts[0],
        "display_name": parts[1],
        "gguf_path": parts[2],
        "tokenizer_id": parts[3],
        "max_context": int(parts[4]),
        "tool_parser": parts[5],
        "tool_format": parts[6],
        "download_url": parts[7],
        "extended_context": int(parts[8]),
        "description": parts[9],
    }

def load_models() -> list[dict]:
    p = _path()
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        d = _parse_line(line)
        if d:
            out.append(d)
    return out

def get_model_config(key: str) -> dict | None:
    for m in load_models():
        if m["model_key"] == key:
            return m
    return None

def export_model_config(key: str) -> dict:
    """Set os.environ for llama-server and proxy. Return the model dict. Raises if not found."""
    import os
    m = get_model_config(key)
    if not m:
        raise KeyError(f"Model '{key}' not in {_path()}")
    full = (root() / m["gguf_path"]).resolve()
    # Generic (engine-agnostic)
    os.environ["SELECTED_MODEL_KEY"] = m["model_key"]
    os.environ["SELECTED_MODEL_NAME"] = m["display_name"]
    os.environ["MODEL_PATH"] = str(full)
    os.environ["MODEL_TOKENIZER_ID"] = m["tokenizer_id"]
    os.environ["MODEL_MAX_CONTEXT"] = str(m["max_context"])
    os.environ["MODEL_EXTENDED_CONTEXT"] = str(m["extended_context"])
    os.environ["MODEL_TOOL_FORMAT"] = m["tool_format"]
    os.environ["MODEL_DOWNLOAD_URL"] = m["download_url"]
    return m
