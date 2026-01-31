"""Start vLLM OpenAI-compatible server. Uses centralized config and models."""
import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root on path when run as -m run.vllm
if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

from stack import config, models
from stack.paths import root
from stack.settings import VLLM_HOST, VLLM_PORT, VLLM_DTYPE, VLLM_CPU_OFFLOAD_GB, VLLM_CUDAGRAPH_MODE

# Global process reference for signal handler
_vllm_process = None

def cleanup_vllm():
    """Properly terminate all vLLM processes including workers."""
    print("\n\n🛑 Shutting down vLLM server...\n")
    
    # Kill the main process first
    if _vllm_process and _vllm_process.poll() is None:
        try:
            _vllm_process.terminate()
            time.sleep(1)
            if _vllm_process.poll() is None:
                _vllm_process.kill()
        except Exception:
            pass
    
    # Kill vLLM API server processes
    subprocess.run(["pkill", "-9", "-f", "vllm.entrypoints"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Kill vLLM worker processes (VLLM::EngineCore, etc.)
    subprocess.run(["pkill", "-9", "-f", "VLLM::"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Kill any remaining vllm processes
    subprocess.run(["pkill", "-9", "-f", "vllm"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    # Check if port is freed
    port = os.environ.get("VLLM_PORT", "8000")
    time.sleep(0.5)
    result = subprocess.run(["lsof", f"-ti:{port}"], 
                          capture_output=True, text=True)
    if result.stdout.strip():
        print(f"⚠️  Port {port} still in use, forcing cleanup...")
        subprocess.run(["bash", "-c", f"lsof -ti:{port} | xargs -r kill -9"],
                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        print(f"✓ Port {port} is free")
    
    print("✓ vLLM server stopped\n")

def signal_handler(signum, frame):
    """Handle SIGINT (CTRL-C) and SIGTERM gracefully."""
    cleanup_vllm()
    sys.exit(0)

def main() -> int:
    global _vllm_process
    
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    ap = argparse.ArgumentParser(description="Start vLLM OpenAI-compatible server")
    ap.add_argument("-m", "--model", help="Override model key (from config/models.conf)")
    ap.add_argument("-p", "--piecewise", action="store_true",
                    help="Use PIECEWISE cudagraph (default is FULL from config/settings.env)")
    ap.add_argument("-k", "--kill", action="store_true",
                    help="Kill any existing vLLM processes before starting")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print full vLLM command and verify args (no server start)")
    args = ap.parse_args()

    # Kill existing vLLM processes if requested
    if args.kill:
        print("\n🛑 Killing existing vLLM processes...\n")
        cleanup_vllm()
        time.sleep(2)
        print("✓ Done. vLLM server stopped.\n")
        return 0

    model = args.model or config.get_current_model()
    if not model:
        print("\n⚠️  No model selected. Run: ./run/run select model\nOr use: -m MODEL_KEY\n")
        return 1

    try:
        m = models.export_model_config(model)
    except KeyError as e:
        print(f"\n{e}\nAvailable: " + ", ".join(x["model_key"] for x in models.load_models()) + "\n")
        return 1

    # Extended context from config/settings.env (CONTEXT_MODE=extended) or env override EXTENDED_CONTEXT_MODE=1
    extended = int(os.environ.get("EXTENDED_CONTEXT_MODE", config.get_extended_context_mode()))
    base_max = int(os.environ["VLLM_MAX_LEN"])
    ext_ctx = os.environ.get("MODEL_EXTENDED_CONTEXT")
    ACTUAL_MAX_LEN = base_max
    SCALE_FACTOR = None

    if extended and ext_ctx and int(ext_ctx) != base_max:
        ACTUAL_MAX_LEN = int(ext_ctx)
        SCALE_FACTOR = ACTUAL_MAX_LEN / base_max
        os.environ["MODEL_MAX_CONTEXT"] = ext_ctx
        print(f"\n🟡 Extended context: {ACTUAL_MAX_LEN} tokens ({SCALE_FACTOR:.1f}x YaRN)")
        print("   ⚠️  ~50–70% slower\n")
    else:
        print(f"\n🟢 Normal context: {ACTUAL_MAX_LEN} tokens\n")

    host = os.environ.get("VLLM_HOST", VLLM_HOST)
    port = os.environ.get("VLLM_PORT", str(VLLM_PORT))
    dtype = os.environ.get("VLLM_DTYPE", VLLM_DTYPE)
    cpu_offload = os.environ.get("VLLM_CPU_OFFLOAD_GB", str(VLLM_CPU_OFFLOAD_GB))

    argv = [
        "--model", os.environ["VLLM_GGUF_MODEL"],
        "--tokenizer", os.environ["VLLM_TOKENIZER_ID"],
        "--served-model-name", model,
        "--host", host,
        "--port", port,
        "--dtype", dtype,
        "--max-model-len", str(ACTUAL_MAX_LEN),
        "--tensor-parallel-size", "1",
    ]

    # Cudagraph: default from config/settings.env (FULL); -p forces PIECEWISE
    use_full_cudagraph = not args.piecewise and (VLLM_CUDAGRAPH_MODE.upper() == "FULL")

    if not use_full_cudagraph:
        argv += ["--compilation-config", '{"cudagraph_mode": "PIECEWISE"}']
        print("Cudagraph: PIECEWISE\n")
    else:
        print("Cudagraph: FULL\n")

    if SCALE_FACTOR is not None:
        # vLLM 0.14+ has no --rope-scaling; use --hf-overrides (rope_parameters) for YaRN
        # No --cpu-offload-gb: RTX 5090 32GB fits Qwen3-30B MoE + 128K in VRAM (see Hardware Corner
        # benchmarks). vLLM's CPU offload path hits "CPU tensor must be pinned" in 0.14.1.
        hf_overrides = (
            '{"rope_parameters":{'
            f'"rope_type":"yarn","factor":{SCALE_FACTOR},'
            f'"original_max_position_embeddings":{base_max}'
            "}}"
        )
        argv += [
            "--hf-overrides", hf_overrides,
            "--kv-cache-dtype", "fp8",
            "--gpu-memory-utilization", "0.90",
        ]
        print("   GPU-only (no CPU offload); reduce --max-model-len if OOM.\n")

    tool_parser = os.environ.get("VLLM_TOOL_PARSER", "none")
    if tool_parser != "none":
        argv += ["--enable-auto-tool-choice", "--tool-call-parser", tool_parser]
        print(f"Tool calling: ENABLED (parser: {tool_parser})\n")
    else:
        print("Tool calling: DISABLED\n")

    log_dir = root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vllm-server.log"
    with open(log_file, "w") as f:
        f.write("")

    print(f"Model: {m['display_name']}")
    print(f"Logs: {log_file}\n")

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Use the installed vllm from site-packages. Do not set PYTHONPATH to vllm source.

    cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve"] + argv

    if args.dry_run:
        # Print full command (shell-quoted for copy-paste) and verify vLLM accepts args
        from shlex import join as shlex_join
        print("Full vLLM command (copy-paste to run manually):\n")
        print(shlex_join(cmd))
        print("\nVerifying vLLM accepts these arguments...")
        proc = subprocess.Popen(
            cmd, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True
        )
        try:
            _, stderr = proc.communicate(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            stderr = ""
        if stderr and "unrecognized arguments" in stderr:
            print("ERROR: vLLM rejected some arguments:\n")
            print(stderr[:2000])
            return 1
        print("OK: vLLM accepted arguments (no 'unrecognized arguments' error).\n")
        return 0

    try:
        with open(log_file, "ab") as lf:
            _vllm_process = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)
            print("Press CTRL-C to stop the server gracefully.\n")
            returncode = _vllm_process.wait()
            return returncode
    except KeyboardInterrupt:
        cleanup_vllm()
        return 0
    except Exception as e:
        print(f"Error starting vLLM: {e}")
        cleanup_vllm()
        return 1

if __name__ == "__main__":
    sys.exit(main())
