"""Start vision API server."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Parse args FIRST to set DEBUG before any imports
_ap = argparse.ArgumentParser(description="Start vision API server")
_ap.add_argument("-m", "--model", help="Override vision model key (from config/vision-models.conf)")
_ap.add_argument("--host", default="0.0.0.0", help="Host to bind to")
_ap.add_argument("--port", default="8004", help="Port to bind to")
_ap.add_argument("--llamacpp-bin", help="Path to llama.cpp vision binary")
_ap.add_argument("-d", "--debug", action="store_true", help="Enable debug mode")
_ap.add_argument("-k", "--kill", action="store_true", help="Kill any existing vision processes before starting")
_args = _ap.parse_args()

# Set DEBUG environment variable BEFORE any imports
os.environ["DEBUG"] = "1" if _args.debug else "0"

# Ensure project root on path
if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

# NOW import (after DEBUG is set)
from stack import config
from stack.vision_models import export_vision_model_config, is_vision_model_downloaded
from stack.paths import root
from stack.settings import VISION_HOST, VISION_PORT


def cleanup_vision():
    """Kill any existing vision API processes."""
    print("Stopping vision API processes...")
    subprocess.run(["pkill", "-9", "-f", "vision.server"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "llama-mtmd-cli"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["pkill", "-9", "-f", "llama-llava-cli"], 
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print("✓ Vision processes stopped")


def main() -> int:
    args = _args  # Use pre-parsed args
    
    # Kill existing vision processes if requested
    if args.kill:
        print("\n🛑 Killing existing vision processes...\n")
        cleanup_vision()
        import time
        time.sleep(1)
        print("✓ Done. Vision server stopped.\n")
        return 0
    
    # Get vision model (from arg or config)
    try:
        vision_model = args.model or config.get_config().get("vision_model", "qwen2-vl-2b-q4")
    except:
        vision_model = args.model or "qwen2-vl-2b-q4"
    
    if not vision_model:
        print("\n⚠️  No vision model selected.")
        print("Available models:")
        from stack.vision_models import load_vision_models
        for m in load_vision_models():
            indicator = "✓" if is_vision_model_downloaded(m["model_key"]) else "✗"
            print(f"  {indicator} {m['model_key']:<20} - {m['display_name']}")
        print("\nSelect with: ./run/select_vision_model.sh")
        print("Or use: -m MODEL_KEY\n")
        return 1
    
    # Check if model is downloaded
    if not is_vision_model_downloaded(vision_model):
        print(f"\n⚠️  Vision model '{vision_model}' not downloaded.")
        print("Download with: ./stack/download_vision_model.sh {vision_model}\n")
        return 1
    
    # Export model config to environment
    try:
        env_vars = export_vision_model_config(vision_model)
        for key, value in env_vars.items():
            os.environ[key] = value
    except KeyError as e:
        print(f"\n{e}")
        return 1
    
    # Set llama.cpp binary path
    if args.llamacpp_bin:
        os.environ["LLAMACPP_BIN"] = args.llamacpp_bin
    else:
        # Default: look in external/llama.cpp/build/bin/
        # Use llama-mtmd-cli (all specialized binaries are deprecated and redirect to this)
        default_bin = root() / "external" / "llama.cpp" / "build" / "bin" / "llama-mtmd-cli"
        
        if default_bin.exists():
            os.environ["LLAMACPP_BIN"] = str(default_bin)
        else:
            print(f"\n⚠️  llama.cpp multimodal binary not found: {default_bin}")
            print("Build llama.cpp with: ./setup/build_llamacpp.sh")
            print("Or use --llamacpp-bin to specify a custom path\n")
            return 1
    
    # Setup logging
    log_dir = root() / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "vision-api.log"
    
    # Clear log file
    with open(log_file, "w") as f:
        f.write("")
    
    print(f"\n🚀 Starting Vision API Server")
    print(f"   Model:      {os.environ.get('VISION_DISPLAY_NAME', vision_model)}")
    print(f"   Listen:     {args.host}:{args.port}")
    print(f"   GGUF:       {os.path.basename(os.environ['VISION_GGUF_PATH'])}")
    print(f"   MMProj:     {os.path.basename(os.environ['VISION_MMPROJ_PATH'])}")
    print(f"   llama.cpp:  {os.environ['LLAMACPP_BIN']}")
    print(f"   Debug:      {'On' if args.debug else 'Off'}")
    print(f"   Logs:       {log_file}\n")
    print("⚠️  Note: Vision runs on CPU - slower than GPU-based text inference")
    print("   Expect 30-60 seconds per image analysis\n")
    
    # Setup logging
    if not args.debug:
        # Production: logs only to file
        sys.stdout.flush()
        sys.stderr.flush()
        
        log_fd = open(log_file, "a", buffering=1)
        os.dup2(log_fd.fileno(), sys.stdout.fileno())
        os.dup2(log_fd.fileno(), sys.stderr.fileno())
    else:
        # Debug: tee output to both console and file
        print("   Debug mode: Output shown in console + log file\n")
        
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
        "vision.server:app",
        host=args.host,
        port=int(args.port),
        log_level="debug" if args.debug else "info",
        access_log=args.debug
    )
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
