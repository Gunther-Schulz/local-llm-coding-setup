# run/ – Single entry point

One main conda env (from `./setup/install.sh`). All commands via **./run/run** from project root.

## Usage

```bash
./run/run llm              # Start LLM backend (llama-server)
./run/run proxy            # Start compression proxy
./run/run vision           # Start vision API
./run/run select model     # Pick LLM model
./run/run select vision    # Pick vision model
```

## Recommended flow

1. `./run/run select model` – choose LLM model  
2. `./run/run llm` – start LLM backend (llama-server)  
3. `./run/run proxy` – start proxy (in another terminal)  
4. Optional: `./run/run vision` for vision

## Layout

- **run** – Single script; dispatches to Python modules.
- **llm.py** – Starts llama-server (llamacpp.py).
- **llamacpp.py** – LLM server launcher (llama-server).
- **proxy.py**, **vision.py** – Proxy and vision servers.
- **select_model.py**, **select_vision_model.py** – Config selectors.
