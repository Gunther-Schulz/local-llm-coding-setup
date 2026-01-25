"""Vision model configuration and management."""
from pathlib import Path
from typing import List, Dict, Optional

from stack.paths import root


def load_vision_models() -> List[Dict]:
    """Load vision models from config file."""
    conf_file = root() / "config" / "vision-models.conf"
    if not conf_file.exists():
        return []
    
    models = []
    with open(conf_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            
            parts = line.split("|")
            if len(parts) < 9:
                continue
            
            models.append({
                "model_key": parts[0],
                "display_name": parts[1],
                "gguf_path": parts[2],
                "mmproj_path": parts[3],
                "max_context": int(parts[4]),
                "quantization": parts[5],
                "download_url_model": parts[6],
                "download_url_mmproj": parts[7],
                "ram_usage": parts[8],
                "capabilities": parts[9] if len(parts) > 9 else ""
            })
    
    return models


def get_vision_model(model_key: str) -> Optional[Dict]:
    """Get vision model config by key."""
    models = load_vision_models()
    for model in models:
        if model["model_key"] == model_key:
            return model
    return None


def export_vision_model_config(model_key: str) -> Dict[str, str]:
    """
    Export vision model config as environment variables dict.
    
    Returns dict that can be used to set os.environ.
    """
    model = get_vision_model(model_key)
    if not model:
        available = ", ".join(m["model_key"] for m in load_vision_models())
        raise KeyError(f"Vision model '{model_key}' not found. Available: {available}")
    
    # Resolve paths relative to root
    root_path = root()
    gguf_path = root_path / model["gguf_path"]
    mmproj_path = root_path / model["mmproj_path"]
    
    return {
        "VISION_MODEL_KEY": model_key,
        "VISION_GGUF_PATH": str(gguf_path),
        "VISION_MMPROJ_PATH": str(mmproj_path),
        "VISION_MAX_CONTEXT": str(model["max_context"]),
        "VISION_DISPLAY_NAME": model["display_name"]
    }


def is_vision_model_downloaded(model_key: str) -> bool:
    """Check if vision model files exist locally."""
    model = get_vision_model(model_key)
    if not model:
        return False
    
    root_path = root()
    gguf_path = root_path / model["gguf_path"]
    mmproj_path = root_path / model["mmproj_path"]
    
    return gguf_path.exists() and mmproj_path.exists()
