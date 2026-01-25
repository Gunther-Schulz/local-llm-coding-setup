"""Single download: aria2c > hf-cli > wget > curl. Used by model-select and later vision."""
import os
import shutil
import subprocess
from pathlib import Path

def download_file(url: str, dest: str | Path, min_bytes: int = 0) -> bool:
    """
    Returns True if file exists and is complete (or was successfully downloaded).
    dest: path to the file to create.
    min_bytes: if >0 and file exists, consider complete only when size >= min_bytes and no .aria2.
    """
    dest = Path(dest)
    a2 = dest.parent / (dest.name + ".aria2")
    if dest.exists() and not a2.exists():
        if min_bytes <= 0:
            return True
        try:
            if dest.stat().st_size >= min_bytes:
                return True
        except OSError:
            pass
        dest.unlink(missing_ok=True)

    if not url or url.lower() == "none":
        return False

    dest.parent.mkdir(parents=True, exist_ok=True)
    d, out = dest.parent, dest.name

    # aria2c
    if _which("aria2c"):
        rc = subprocess.run(
            ["aria2c", "--continue=true", "--max-connection-per-server=16",
             "--min-split-size=1M", "--split=16", "--file-allocation=none",
             "--console-log-level=warn", "--summary-interval=0",
             "-d", str(d), "-o", out, url],
            capture_output=True, cwd=str(d),
        )
        a2.unlink(missing_ok=True)
        if dest.exists():
            return True

    # huggingface-cli (only for HF URLs)
    if _which("huggingface-cli") and "huggingface.co" in url:
        import re
        m = re.match(r"https?://huggingface\.co/([^/]+/[^/]+)/resolve/[^/]+/(.+)$", url)
        if m:
            repo, filename = m.group(1), m.group(2)
            env = os.environ.copy()
            env.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
            rc = subprocess.run(
                ["huggingface-cli", "download", repo, filename, "--local-dir", str(d), "--local-dir-use-symlinks", "False"],
                env=env, capture_output=True,
            )
            f = d / filename
            if f.exists() and f != dest:
                f.rename(dest)
            if dest.exists():
                return True

    # wget
    if _which("wget"):
        rc = subprocess.run(["wget", "-c", "-O", str(dest), url], capture_output=True)
        if dest.exists():
            return True

    # curl
    if _which("curl"):
        rc = subprocess.run(["curl", "-L", "-C", "-", "-o", str(dest), url], capture_output=True)
        if dest.exists():
            return True

    return dest.exists()

def _which(name: str) -> bool:
    return shutil.which(name) is not None
