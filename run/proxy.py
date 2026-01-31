"""Start compression proxy server."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Parse args first (don't set DEBUG yet – config may override)
_ap = argparse.ArgumentParser(description="Start compression proxy server")
_ap.add_argument("--host", default="0.0.0.0", help="Host to bind to")
_ap.add_argument("--port", default="8002", help="Port to bind to")
_ap.add_argument("--vllm-url", default="http://localhost:8000", help="vLLM backend URL")
_ap.add_argument("--vision-url", default="http://localhost:8004", help="Vision API URL")
_ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode (or set DEBUG=1 in config/settings.env)")
_ap.add_argument("-k", "--kill", action="store_true", help="Kill any existing proxy processes before starting")
_args = _ap.parse_args()

# Ensure project root on path before loading config
if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

# Load config/settings.env first so DEBUG=1 there is respected
from stack import settings as _stack_settings  # noqa: F401 – loads config/settings.env
os.environ["VLLM_URL"] = _args.vllm_url
os.environ["VISION_API_URL"] = _args.vision_url
# Debug: from -d/--debug or from config/settings.env DEBUG=1
os.environ["DEBUG"] = "1" if (_args.debug or os.getenv("DEBUG", "0") == "1") else "0"

from stack import config
from stack.paths import root
from stack.settings import PROXY_HOST, PROXY_PORT, VLLM_URL, VISION_URL


def cleanup_proxy():
    """Kill any existing proxy processes."""
    print("Stopping proxy processes...")
    subprocess.run(["pkill", "-9", "-f", "proxy.server"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ Proxy processes stopped")


def main() -> int:
    args = _args  # Use pre-parsed args
    
    # Kill existing proxy processes if requested
    if args.kill:
        print("\n🛑 Killing existing proxy processes...\n")
        cleanup_proxy()
        import time
        time.sleep(1)
        print("✓ Done. Proxy stopped.\n")
        return 0
    
    # Get model context from config (same as vLLM launcher)
    model = config.get_current_model()
    if model:
        try:
            from stack import models
            from stack.settings import get_effective_context_limit
            model_info = models.export_model_config(model)
            # Use extended (128K) only when context mode is "extended" (YaRN), else base (32K)
            if config.get_context_mode() != "extended":
                os.environ["MODEL_EXTENDED_CONTEXT"] = ""
            effective_ctx = get_effective_context_limit()
            os.environ["MODEL_MAX_CONTEXT"] = str(effective_ctx)
            os.environ["MODEL_TOOL_FORMAT"] = model_info.get("tool_format", "openai")
        except Exception as e:
            print(f"⚠️  Warning: Could not load model config: {e}")
    # When CONTEXT_MODE=extended but no model selected (or model didn't set it), use 128K so proxy matches backend
    if config.get_context_mode() == "extended" and not os.environ.get("MODEL_EXTENDED_CONTEXT"):
        default_extended = os.getenv("MODEL_EXTENDED_CONTEXT_DEFAULT", "131072")
        os.environ["MODEL_EXTENDED_CONTEXT"] = default_extended
        os.environ["MODEL_MAX_CONTEXT"] = default_extended
    
    # Setup logging
    log_dir = root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "proxy.log"
    
    # Clear log file
    with open(log_file, "w") as f:
        f.write("")
    
    effective_ctx = int(os.environ.get("MODEL_MAX_CONTEXT", "32768"))
    comp = "Cursor-style on overflow" if os.getenv("COMPRESSION_ENABLED", "0") == "1" else "off"
    print(f"\n🚀 Starting Compression Proxy")
    print(f"   Listen:     {args.host}:{args.port}")
    print(f"   vLLM:       {args.vllm_url}")
    print(f"   Vision:     {args.vision_url}")
    print(f"   Context:    {effective_ctx} tokens (compression: {comp})")
    debug_on = os.getenv("DEBUG", "0") == "1"
    print(f"   Debug:      {'On' if debug_on else 'Off'} (-d or DEBUG=1 in config/settings.env)")
    print(f"   Logs:       {log_file}\n")
    
    # Setup logging: Tee to console + file when debug so [DEBUG] flow/messages go to log
    if not debug_on:
        # Production: logs only to file
        sys.stdout.flush()
        sys.stderr.flush()
        
        log_fd = open(log_file, "a", buffering=1)
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
    else:
        # Debug: tee output to both console and file (full request/response flow in log)
        print("   Debug mode: Output shown in console + log file (request/response flow)\n")
        
        class TeeOutput:
            """Write to both console and file."""
            def __init__(self, console, log_file):
                self.console = console
                self.log_file = open(log_file, "a", buffering=1)
            
            def write(self, message):
                self.console.write(message)
                self.log_file.write(message)
            
            def flush(self):
                self.console.flush()
                self.log_file.flush()
            
            def isatty(self):
                """Check if console is a TTY."""
                return self.console.isatty()
        
        sys.stdout = TeeOutput(sys.__stdout__, log_file)
        sys.stderr = TeeOutput(sys.__stderr__, log_file)
    
    # Import and run uvicorn
    import uvicorn
    
    uvicorn.run(
        "proxy.server:app",
        host=args.host,
        port=int(args.port),
        log_level="debug" if debug_on else "info",
        access_log=debug_on
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
