"""Start the LLM backend (llama-server). Uses central config and models."""
import os
import sys
from pathlib import Path

if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

from stack import config, models


def _model_from_argv():
    """Parse -m / --model from sys.argv so we export the right model before dispatching."""
    for i, a in enumerate(sys.argv):
        if a in ("-m", "--model") and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return None


def main() -> int:
    model = _model_from_argv() or config.get_current_model()

    if not model:
        print("\n⚠️  No model selected. Run: ./run/run select model\nOr use: -m MODEL_KEY\n")
        return 1

    os.environ["LLM_MODEL"] = model
    try:
        models.export_model_config(model)
    except KeyError as e:
        print(f"\n{e}\nAvailable: " + ", ".join(x["model_key"] for x in models.load_models()) + "\n")
        return 1

    from run.llamacpp import main as llamacpp_main
    return llamacpp_main()


if __name__ == "__main__":
    sys.exit(main())
