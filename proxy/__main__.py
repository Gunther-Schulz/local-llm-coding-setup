"""Run the proxy server. Usage: python -m proxy [--debug]"""
from __future__ import annotations

import sys

from .config import load_config
from .server import run_server


def main() -> None:
    debug = "--debug" in sys.argv
    config = load_config()
    run_server(config, debug=debug)


if __name__ == "__main__":
    main()
