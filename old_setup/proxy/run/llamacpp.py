"""Start llama-server (llama.cpp) OpenAI-compatible server. Uses central config and models."""
import os
import signal
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

from stack import config, models
from stack.paths import root
from stack.settings import (
    VLLM_HOST,
    VLLM_PORT,
    LLAMACPP_SERVER_BIN,
    FIT_CONTEXT,
    CACHE_TYPE_K,
    MOE_OFFLOAD_REGEX,
    LLAMACPP_TEMP,
    LLAMACPP_TOP_P,
    LLAMACPP_TOP_K,
    LLAMACPP_MIN_P,
    LLAMACPP_SEED,
)

_process = None


def cleanup():
    global _process
    if _process and _process.poll() is None:
        try:
            _process.terminate()
            _process.wait(timeout=5)
        except Exception:
            _process.kill()
    _process = None


def _signal_handler(signum, frame):
    cleanup()
    sys.exit(0)


def main() -> int:
    global _process
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    model = os.environ.get("LLM_MODEL") or config.get_current_model()
    if not model:
        print("\n⚠️  No model selected. Run: ./run/run select model\nOr use: -m MODEL_KEY\n")
        return 1

    try:
        m = models.export_model_config(model)
    except KeyError as e:
        print(f"\n{e}\nAvailable: " + ", ".join(x["model_key"] for x in models.load_models()) + "\n")
        return 1

    model_path = os.environ.get("MODEL_PATH")
    if not model_path or not Path(model_path).exists():
        print(f"\n⚠️  Model file not found: {model_path}\n")
        return 1

    # Context: extended from config/settings.env (CONTEXT_MODE) or base from model
    extended = config.get_context_mode() == "extended"
    ext_ctx = os.environ.get("MODEL_EXTENDED_CONTEXT")
    max_ctx = int(ext_ctx) if (extended and ext_ctx) else int(os.environ.get("MODEL_MAX_CONTEXT", "32768"))

    # Per-model llama-server options (from models.conf cols 15–17) else config/llamacpp.env
    llm_cfg = models.get_model_llamacpp_config(model)
    fit = llm_cfg.get("fit_context") if llm_cfg.get("fit_context") is not None else FIT_CONTEXT
    cache_k = llm_cfg.get("cache_type_k") or CACHE_TYPE_K
    moe_off = llm_cfg.get("moe_offload")
    if moe_off == "moe":
        moe_regex = ".ffn_.*_exps.=CPU"
    else:
        moe_regex = (moe_off or MOE_OFFLOAD_REGEX) if moe_off else MOE_OFFLOAD_REGEX
    temp = LLAMACPP_TEMP
    top_p = LLAMACPP_TOP_P
    top_k = LLAMACPP_TOP_K
    min_p = LLAMACPP_MIN_P
    seed = LLAMACPP_SEED

    server_bin = Path(LLAMACPP_SERVER_BIN)
    if not server_bin.is_absolute():
        server_bin = (root() / LLAMACPP_SERVER_BIN).resolve()
    if not server_bin.exists() or not os.access(server_bin, os.X_OK):
        print(f"\n⚠️  llama-server not found: {server_bin}")
        print("   Run ./setup/install.sh (includes llama.cpp CUDA) or ./setup/build/llamacpp_cuda.sh")
        print("   Or set LLAMACPP_SERVER_BIN=/path/to/llama-server\n")
        return 1

    host = os.environ.get("VLLM_HOST", VLLM_HOST)
    port = os.environ.get("VLLM_PORT", str(VLLM_PORT))

    # --jinja required for OpenAI-style function/tool calling (see docs/function-calling.md)
    argv = [str(server_bin), "-m", model_path, "--host", host, "--port", port, "--n-gpu-layers", "-1", "--jinja"]
    if fit:
        argv.append("--fit")
        argv.append("on")
    else:
        argv.extend(["-c", str(max_ctx)])
    if cache_k:
        argv.extend(["--cache-type-k", cache_k])
    if moe_regex:
        argv.extend(["-ot", moe_regex])
    if temp:
        argv.extend(["--temp", temp])
    if top_p:
        argv.extend(["--top-p", top_p])
    if top_k:
        argv.extend(["--top-k", top_k])
    if min_p:
        argv.extend(["--min-p", min_p])
    if seed:
        argv.extend(["--seed", seed])

    log_dir = root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "llamacpp-server.log"
    log_file.write_text("")

    print(f"Model: {m['display_name']}")
    print(f"Context: {'fit on' if fit else f'{max_ctx} tokens'}")
    print(f"Backend: http://{host}:{port}")
    print(f"Logs: {log_file}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"

    try:
        with open(log_file, "ab") as lf:
            _process = subprocess.Popen(argv, env=env, stdout=lf, stderr=subprocess.STDOUT, cwd=str(root()))
        print("Press CTRL-C to stop the server.\n")
        return _process.wait()
    except KeyboardInterrupt:
        cleanup()
        return 0
    except Exception as e:
        print(f"Error starting llama-server: {e}")
        cleanup()
        return 1


if __name__ == "__main__":
    sys.exit(main())
