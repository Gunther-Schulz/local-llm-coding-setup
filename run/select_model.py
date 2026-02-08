"""Interactive model selection. Writes to config/llm-config. Uses stack.config and stack.models."""
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

from stack import config, models
from stack.download import download_file
from stack.paths import root

MIN_GGUF_BYTES = 1024 * 1024 * 1024  # 1 GiB

def main() -> int:
    config.migrate_from_dotfiles()

    current = config.get_current_model()
    if current:
        r = input(f"Current model: {current}. Change? [y/N]: ").strip().lower()
        if r != "y":
            print("Keeping current model.")
            return 0

    lst = models.load_models()
    if not lst:
        print("No models in config/models.conf")
        return 1

    print("\n" + "═" * 60 + "\n  📦 LLM Model Selection\n" + "═" * 60 + "\n")

    for i, m in enumerate(lst, 1):
        p = root() / m["gguf_path"]
        status = "✓" if p.exists() and (p.stat().st_size >= MIN_GGUF_BYTES) else "✗"
        print(f"  [{i}] {status}  {m['display_name']}")
        print(f"      Context: {m['max_context']} | Tool format: {m['tool_format']}")
        print(f"      {m['description']}\n")

    print("═" * 60 + "\n")

    while True:
        sel = input(f"Select [1–{len(lst)}] or 'q' to quit: ").strip().lower()
        if sel == "q":
            print("Cancelled.")
            return 1

        try:
            idx = int(sel)
            if 1 <= idx <= len(lst):
                break
        except ValueError:
            pass
        print(f"Invalid. Enter 1–{len(lst)} or q.")

    m = lst[idx - 1]
    path = root() / m["gguf_path"]

    if not path.exists() or path.stat().st_size < MIN_GGUF_BYTES:
        url = m["download_url"]
        if not url or url.lower() == "none":
            print(f"No download URL for {m['model_key']}. Please add the file to:\n  {path}")
            return 1
        r = input("Model not downloaded. Download now? [Y/n]: ").strip().lower()
        if r == "n":
            print("Please pick a model that is already on disk.")
            return 1
        print(f"\n📥 Downloading: {m['display_name']} …\n")
        if not download_file(url, path, MIN_GGUF_BYTES):
            print("Download failed.")
            return 1
        print("✅ Download complete.\n")

    config.set_current_model(m["model_key"])
    models.export_model_config(m["model_key"])

    print("═" * 60)
    print(f"  ✅ Model: {m['display_name']}")
    print(f"  Context: {os.environ.get('MODEL_MAX_CONTEXT')} tokens | Extended: {os.environ.get('MODEL_EXTENDED_CONTEXT', '—')} | Format: {os.environ.get('MODEL_TOOL_FORMAT')}")
    print("═" * 60)
    print("\nStart LLM:   ./run/run llm")
    print("Override:    ./run/run llm -m MODEL_KEY\n")

    return 0

if __name__ == "__main__":
    sys.exit(main())
