# run/ – Single entry point

One main conda env (from `./setup/install.sh`). All commands via **./run/run** from project root.

## Usage

```bash
./run/run llm              # Start LLM backend (engine from config: vLLM or llama.cpp)
./run/run proxy            # Start compression proxy
./run/run vision           # Start vision API
./run/run select model     # Pick LLM model
./run/run select engine    # Pick backend: vllm | llamacpp (e.g. ./run/run select engine llamacpp)
./run/run select vision   # Pick vision model
```

## Recommended flow

1. `./run/run select model` – choose LLM model  
2. `./run/run select engine` [vllm|llamacpp] – choose backend (default: vllm)  
3. `./run/run llm` – start LLM backend  
4. `./run/run proxy` – start proxy (in another terminal)  
5. Optional: `./run/run vision` for vision

## Layout

- **run** – Single script; dispatches to Python modules.
- **llm.py** – Dispatcher: reads engine from config, runs vllm or llamacpp.
- **vllm.py**, **llamacpp.py** – LLM server launchers.
- **proxy.py**, **vision.py** – Proxy and vision servers.
- **select_model.py**, **select_engine.py**, **select_vision_model.py** – Config selectors.
