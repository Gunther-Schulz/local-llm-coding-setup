"""
Config loader for tool proxy.
Loads and validates YAML configuration files.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration loading error."""
    pass


def load_rules(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load rules from YAML config file.
    
    Args:
        config_path: Path to YAML config file. If None, uses default_rules.yaml
        
    Returns:
        Dict of configured rules
        
    Raises:
        ConfigError: If config cannot be loaded or validated
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent / "default_rules.yaml"
    else:
        config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_rules()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in {config_path}: {e}")
    except Exception as e:
        raise ConfigError(f"Error reading {config_path}: {e}")
    
    if config is None:
        logger.warning(f"Empty config file: {config_path}, using defaults")
        return get_default_rules()
    
    # Validate schema
    validate_config(config)
    
    # Merge with defaults for any missing keys
    return merge_with_defaults(config)


def get_default_rules() -> Dict[str, Any]:
    """Return default configuration rules."""
    return {
        "tools": {
            "Read": {
                "enabled": True,
                "message": "REMINDER: Use Grep/Glob first for discovery. Read only after you know what file(s) to examine."
            },
            "Grep": {
                "enabled": True,
                "message": "REMINDER: Grep is for finding patterns across files. Use Glob to discover files first."
            },
            "Glob": {
                "enabled": True,
                "message": "REMINDER: Glob discovers files by pattern. Use Grep to search file contents."
            },
            "LS": {
                "enabled": True,
                "message": "REMINDER: LS lists directory contents. Use Glob for pattern-based file discovery."
            }
        },
        "read_coalescing": {
            "enabled": True,
            "max_reads_per_turn": 3,
            "reminder_message": "WARNING: This file/range has been read multiple times this turn. Avoid loops!"
        },
        "overlapping_ranges": {
            "enabled": True,
            "max_overlapping_reads": 2,
            "reminder_message": "WARNING: Multiple overlapping reads detected. Consider consolidating."
        },
        "turn_tracking": {
            "enabled": True,
            "max_turns_in_memory": 100,
            "auto_reset_turn": True
        },
        "logging": {
            "level": "INFO",
            "file": None,
            "log_tool_calls": False,
            "log_reminders": True,
            "debug": False
        }
    }


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration schema.
    
    Args:
        config: Loaded configuration dict
        
    Raises:
        ConfigError: If validation fails
    """
    required_keys = ["tools", "read_coalescing", "overlapping_ranges", "turn_tracking", "logging"]
    
    for key in required_keys:
        if key not in config:
            raise ConfigError(f"Missing required config key: {key}")
    
    # Validate tools
    if not isinstance(config["tools"], dict):
        raise ConfigError("tools must be a dict")
    
    # Validate read_coalescing
    rc = config["read_coalescing"]
    if not isinstance(rc, dict):
        raise ConfigError("read_coalescing must be a dict")
    if "enabled" not in rc:
        rc["enabled"] = True
    if "max_reads_per_turn" not in rc:
        rc["max_reads_per_turn"] = 3
    
    # Validate logging
    logging_config = config["logging"]
    if not isinstance(logging_config, dict):
        raise ConfigError("logging must be a dict")
    if "level" not in logging_config:
        logging_config["level"] = "INFO"


def merge_with_defaults(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge user config with defaults for missing keys.
    
    Args:
        config: User-provided configuration
        
    Returns:
        Merged configuration dict
    """
    defaults = get_default_rules()
    
    # Deep merge
    result = {}
    
    for key in defaults:
        if key in config:
            if isinstance(defaults[key], dict) and isinstance(config[key], dict):
                # Merge nested dicts
                merged = defaults[key].copy()
                merged.update(config[key])
                result[key] = merged
            else:
                result[key] = config[key]
        else:
            result[key] = defaults[key]
    
    return result