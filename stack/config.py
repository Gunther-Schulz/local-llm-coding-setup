"""Runtime config: current model from llm-config; context mode from config/settings.env (CONTEXT_MODE) or llm-config."""
import os
import configparser
from pathlib import Path
from typing import Dict, Optional, Any
from .paths import root

CONFIG_NAME = "config/llm-config"

def _path() -> Path:
    return root() / CONFIG_NAME

def _ensure_config_dir() -> None:
    _path().parent.mkdir(parents=True, exist_ok=True)

def init_config() -> None:
    p = _path()
    if p.exists():
        return
    _ensure_config_dir()
    p.write_text("""# LLM Configuration
# Managed by select-model. Do not edit manually.

[model]
key =
selected_at =

[context]
mode = normal
# Overridden by config/settings.env CONTEXT_MODE (normal | extended)
# extended = YaRN 128K; normal = 32K

[engine]
# Backend to run: vllm | llamacpp (uses same models.conf and port 8000)
key = vllm
""")

def _read(section: str, key: str) -> str:
    cp = configparser.ConfigParser()
    p = _path()
    if not p.exists():
        return ""
    cp.read(p)
    try:
        return (cp.get(section, key) or "").strip()
    except (configparser.NoSectionError, configparser.NoOptionError):
        return ""

def _write(section: str, key: str, value: str) -> None:
    init_config()
    cp = configparser.ConfigParser()
    cp.read(_path())
    if not cp.has_section(section):
        cp.add_section(section)
    cp.set(section, key, value)
    with open(_path(), "w") as f:
        cp.write(f)

def get_current_model() -> str:
    return _read("model", "key")

def set_current_model(key: str) -> None:
    from datetime import datetime, timezone
    _write("model", "key", key)
    _write("model", "selected_at", datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"))

def get_context_mode() -> str:
    # Ensure config/settings.env is loaded so CONTEXT_MODE is in os.environ
    try:
        from stack import settings  # noqa: F401 - loads settings.env
    except Exception:
        pass
    # config/settings.env CONTEXT_MODE is the central switch (normal | extended)
    env_mode = os.environ.get("CONTEXT_MODE", "").strip().lower()
    if env_mode in ("normal", "extended"):
        return env_mode
    return _read("context", "mode") or "normal"

def set_context_mode(mode: str) -> None:
    if mode not in ("normal", "extended"):
        raise ValueError(f"Invalid context mode: {mode}")
    _write("context", "mode", mode)

def get_extended_context_mode() -> int:
    return 1 if get_context_mode() == "extended" else 0

def set_extended_context_mode(value: int | str) -> None:
    set_context_mode("extended" if str(value) == "1" else "normal")

def get_engine() -> str:
    """Current LLM engine: vllm | llamacpp. Env LLM_ENGINE overrides."""
    env_engine = os.environ.get("LLM_ENGINE", "").strip().lower()
    if env_engine in ("vllm", "llamacpp"):
        return env_engine
    return _read("engine", "key") or "vllm"

def set_engine(engine: str) -> None:
    if engine not in ("vllm", "llamacpp"):
        raise ValueError(f"Invalid engine: {engine}")
    _write("engine", "key", engine)

def get_config() -> dict:
    """Get all config values as a dict."""
    cp = configparser.ConfigParser()
    p = _path()
    if not p.exists():
        return {}
    cp.read(p)
    
    result = {}
    for section in cp.sections():
        for key, value in cp.items(section):
            result[f"{section}.{key}"] = value
            # Also add without section prefix for convenience
            if key not in result:
                result[key] = value
    return result


def set_config(key: str, value: str, section: str = "runtime") -> None:
    """Set a config value. Key can be 'section.key' or just 'key' (uses section param)."""
    if "." in key:
        section, key = key.split(".", 1)
    _write(section, key, value)


def migrate_from_dotfiles() -> bool:
    """Migrate .current-model, .context-mode, .llm-config into config/llm-config. Return True if any migrated."""
    r = root()
    migrated = False

    # .current-model -> [model] key
    old_model = r / ".current-model"
    if old_model.exists():
        key = old_model.read_text().splitlines()[0].strip()
        if key:
            set_current_model(key)
            migrated = True
        old_model.rename(r / ".current-model.bak")

    # .context-mode -> [context] mode
    old_ctx = r / ".context-mode"
    if old_ctx.exists():
        val = old_ctx.read_text().splitlines()[0].strip()
        set_extended_context_mode(val)
        migrated = True
        old_ctx.rename(r / ".context-mode.bak")

    # .llm-config at root -> config/llm-config (if new doesn't exist)
    old_ini = r / ".llm-config"
    dst = _path()
    if old_ini.exists() and not dst.exists():
        _ensure_config_dir()
        import shutil
        shutil.copy2(old_ini, dst)
        migrated = True
        old_ini.rename(r / ".llm-config.bak")

    return migrated
