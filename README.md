# RunPod LLM System

A modular system for managing local LLM inference with vLLM and llama.cpp backends.

## Architecture

The system follows a clean architecture pattern with clear separation of concerns:

1. **Configuration Management**: Centralized configuration system in `stack/config_manager.py`
2. **Model Management**: Model definitions and handling in `stack/models.py`
3. **Runtime Components**: 
   - `run/llm.py` - Main LLM engine dispatcher
   - `run/vllm.py` - vLLM backend implementation
   - `run/llamacpp.py` - llama.cpp backend implementation
   - `run/proxy.py` - Proxy server for vision processing

## Configuration

Configuration is handled through `config/llm-config` file with the following sections:

```
[model]
key = 
selected_at = 

[context]
mode = normal

[engine]
key = vllm
```

Environment variables can override configuration:
- `LLM_ENGINE` - Set to `vllm` or `llamacpp`
- `CONTEXT_MODE` - Set to `normal` or `extended`

## Key Features

- Centralized configuration management
- Easy switching between vLLM and llama.cpp backends
- Support for different context modes (normal/extended)
- Clear separation of concerns between components
- Robust error handling and validation

## Getting Started

1. Select a model: `./run/run select model`
2. Start the LLM: `./run/run llm`
3. For vision processing: `./run/run vision`

## Code Structure

- `run/` - Main entry points and execution logic
- `stack/` - Core modules and utilities
- `config/` - Configuration files
- `models/` - Model definitions and management