"""Interactive vision model selector."""
import sys
from pathlib import Path

# Ensure project root on path
if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

from stack import config
from stack.vision_models import load_vision_models, is_vision_model_downloaded


def main() -> int:
    models = load_vision_models()
    
    if not models:
        print("\n⚠️  No vision models configured.")
        print("Check config/vision-models.conf\n")
        return 1
    
    # Check current vision model selection
    try:
        current = config.get_config().get("vision_model", "")
    except:
        current = ""
    
    print("\n" + "=" * 70)
    print("  Vision Model Selection (CPU-based via llama.cpp)")
    print("=" * 70)
    
    if current:
        current_model = next((m for m in models if m["model_key"] == current), None)
        if current_model:
            print(f"\nCurrent: {current_model['display_name']}")
        else:
            print(f"\nCurrent: {current} (not found in config)")
    else:
        print("\nCurrent: None selected")
    
    print("\nAvailable Vision Models:\n")
    
    for i, model in enumerate(models, 1):
        downloaded = is_vision_model_downloaded(model["model_key"])
        status = "✓ READY" if downloaded else "✗ NOT DOWNLOADED"
        current_marker = "→" if model["model_key"] == current else " "
        
        print(f"{current_marker} {i}. {model['display_name']:<35} [{status}]")
        print(f"     {model['model_key']}")
        print(f"     RAM: {model['ram_usage']:<8} Quant: {model['quantization']}")
        print(f"     {model.get('capabilities', 'Vision understanding')}")
        print()
    
    print("=" * 70)
    
    try:
        choice = input("\nSelect model number (or 'q' to quit): ").strip()
        
        if choice.lower() == 'q':
            return 0
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(models):
                selected = models[idx]
                
                # Check if downloaded
                if not is_vision_model_downloaded(selected["model_key"]):
                    print(f"\n⚠️  Model not downloaded yet.")
                    print(f"Download with:")
                    print(f"  ./stack/download_vision_model.sh {selected['model_key']}")
                    
                    confirm = input("\nSet as active anyway? (y/N): ").strip().lower()
                    if confirm != 'y':
                        return 0
                
                # Save selection
                config.set_config("vision_model", selected["model_key"])
                print(f"\n✓ Vision model set to: {selected['display_name']}")
                print(f"  Start with: ./run/run vision\n")
                return 0
            else:
                print("\n✗ Invalid selection\n")
                return 1
        except ValueError:
            print("\n✗ Invalid input\n")
            return 1
    
    except (KeyboardInterrupt, EOFError):
        print("\n\nCancelled\n")
        return 0


if __name__ == "__main__":
    sys.exit(main())
