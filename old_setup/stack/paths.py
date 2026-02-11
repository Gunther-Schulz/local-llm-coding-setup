"""Resolve project root. Prefer ROOT, WORKSPACE, RUNPOD_ROOT; else parent of stack/."""
from pathlib import Path
import os

def root() -> Path:
    r = os.environ.get("ROOT") or os.environ.get("WORKSPACE") or os.environ.get("RUNPOD_ROOT")
    if r:
        return Path(r).resolve()
    return Path(__file__).resolve().parents[1]
