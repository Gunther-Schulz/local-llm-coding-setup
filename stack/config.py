"""Runtime config: current model from llm-config; context mode from config/settings.env (CONTEXT_MODE) or llm-config."""
from .config_manager import ConfigManager

# Initialize the centralized config manager
_config_manager = ConfigManager()

def init_config() -> None:
    """Initialize default config file if not exists."""
    _config_manager.init_config()

def get_engine() -> str:
    """Current LLM engine: vllm | llamacpp. Env LLM_ENGINE overrides."""
    return _config_manager.get_engine()

def set_engine(engine: str) -> None:
    """Set engine in config."""
    _config_manager.set_engine(engine)

def get_current_model() -> str | None:
    """Get configured model key."""
    return _config_manager.get_current_model()

def set_model(model: str) -> None:
    """Set model in config."""
    _config_manager.set_model(model)

def get_context_mode() -> str:
    """Get context mode (normal or extended)."""
    return _config_manager.get_context_mode()

def set_context_mode(mode: str) -> None:
    """Set context mode."""
    _config_manager.set_context_mode(mode)

def get_config() -> dict:
    """Get all config values as a dict."""
    return _config_manager.get_config()

def set_config(**kwargs) -> None:
    """Set multiple config values."""
    _config_manager.set_config(**kwargs)