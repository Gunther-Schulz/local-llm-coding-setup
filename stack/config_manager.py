"""Centralized configuration management system."""
import os
import configparser
from pathlib import Path
from typing import Optional, Dict, Any
from .paths import root

class ConfigManager:
    """Manages all application configuration in a centralized way."""
    
    def __init__(self, config_file: str = "config/llm-config"):
        self.config_file = root() / config_file
        self._ensure_config_dir()
        
    def _ensure_config_dir(self):
        """Ensure the configuration directory exists."""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        
    def init_config(self) -> None:
        """Initialize default config file if not exists."""
        if self.config_file.exists():
            return
            
        self.config_file.write_text("""# LLM Configuration
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
        
    def _read(self, section: str, key: str) -> str:
        """Read a specific config value."""
        cp = configparser.ConfigParser()
        if not self.config_file.exists():
            return ""
        cp.read(self.config_file)
        try:
            return (cp.get(section, key) or "").strip()
        except (configparser.NoSectionError, configparser.NoOptionError):
            return ""
            
    def _write(self, section: str, key: str, value: str) -> None:
        """Write a specific config value."""
        self.init_config()
        cp = configparser.ConfigParser()
        cp.read(self.config_file)
        if not cp.has_section(section):
            cp.add_section(section)
        cp.set(section, key, value)
        with open(self.config_file, "w") as f:
            cp.write(f)
            
    def get_engine(self) -> str:
        """Get configured engine (vllm or llamacpp). Env LLM_ENGINE overrides."""
        env_engine = os.environ.get("LLM_ENGINE", "").strip().lower()
        if env_engine in ("vllm", "llamacpp"):
            return env_engine
        return self._read("engine", "key") or "vllm"
        
    def set_engine(self, engine: str) -> None:
        """Set engine in config."""
        if engine not in ("vllm", "llamacpp"):
            raise ValueError(f"Invalid engine: {engine}")
        self._write("engine", "key", engine)
        
    def get_current_model(self) -> Optional[str]:
        """Get configured model key."""
        return self._read("model", "key") or None
        
    def set_model(self, model: str) -> None:
        """Set model in config."""
        self._write("model", "key", model)
        
    def get_context_mode(self) -> str:
        """Get context mode (normal or extended)."""
        env_mode = os.environ.get("CONTEXT_MODE", "").strip().lower()
        if env_mode in ("normal", "extended"):
            return env_mode
        return self._read("context", "mode") or "normal"
        
    def set_context_mode(self, mode: str) -> None:
        """Set context mode."""
        if mode not in ("normal", "extended"):
            raise ValueError(f"Invalid context mode: {mode}")
        self._write("context", "mode", mode)
        
    def get_config(self) -> Dict[str, Any]:
        """Get all configuration values as a dictionary."""
        return {
            "engine": self.get_engine(),
            "model": self.get_current_model(),
            "context_mode": self.get_context_mode()
        }
        
    def set_config(self, **kwargs) -> None:
        """Set multiple configuration values at once."""
        for key, value in kwargs.items():
            if key == "engine":
                self.set_engine(value)
            elif key == "model":
                self.set_model(value)
            elif key == "context_mode":
                self.set_context_mode(value)