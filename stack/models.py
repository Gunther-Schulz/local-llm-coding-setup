"""models.conf: registry and export for llama-server/proxy. Single source for model definitions."""
from pathlib import Path
from .paths import root

MODELS_CONF = "config/models.conf"
# Format: 10 cols required; optional cols 11-14: compression, virtual_tool, inject_system, inject_capability (1/0/empty)

def _path() -> Path:
    return root() / MODELS_CONF


def _parse_optional_bool(s: str) -> bool | None:
    """Parse 1/0 or empty; return True/False or None (use global)."""
    if not s or s.strip() in ("", "-"):
        return None
    return s.strip() in ("1", "true", "on", "yes")


def _parse_line(line: str) -> dict | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split("|")
    if len(parts) < 10:
        return None
    d = {
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
    # Optional per-model proxy flags (cols 11-14)
    if len(parts) >= 14:
        d["compression"] = _parse_optional_bool(parts[10])
        d["virtual_tool"] = _parse_optional_bool(parts[11])
        d["inject_system"] = _parse_optional_bool(parts[12])
        d["inject_capability"] = _parse_optional_bool(parts[13])
    else:
        d["compression"] = None
        d["virtual_tool"] = None
        d["inject_system"] = None
        d["inject_capability"] = None
    return d

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

def get_model_proxy_flags(model_key: str) -> dict:
    """
    Return per-model proxy flags for the given model_key.
    Keys: compression, virtual_tool, inject_system, inject_capability.
    Values: True/False or None (use global from settings.env).
    """
    m = get_model_config(model_key)
    if not m:
        return {}
    return {
        "compression": m.get("compression"),
        "virtual_tool": m.get("virtual_tool"),
        "inject_system": m.get("inject_system"),
        "inject_capability": m.get("inject_capability"),
    }


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
