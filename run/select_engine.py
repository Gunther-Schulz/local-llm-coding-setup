"""Set or show the LLM engine (vllm | llamacpp). Uses config/llm-config; override with LLM_ENGINE."""
import sys
from pathlib import Path

if __name__ == "__main__":
    _runpod = Path(__file__).resolve().parents[1]
    if str(_runpod) not in sys.path:
        sys.path.insert(0, str(_runpod))

from stack import config


def main() -> int:
    config.migrate_from_dotfiles()
    current = config.get_engine()

    if len(sys.argv) >= 2:
        engine = sys.argv[1].strip().lower()
        if engine not in ("vllm", "llamacpp"):
            print(f"Invalid engine: {engine}. Use vllm or llamacpp.")
            return 1
        config.set_engine(engine)
        print(f"Engine set to: {engine}\nStart LLM: ./run/run llm\n")
        return 0

    print(f"Current engine: {current}")
    print("To change: ./run/run select engine vllm   or   ./run/run select engine llamacpp")
    print("Or set env: LLM_ENGINE=llamacpp ./run/run llm\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
