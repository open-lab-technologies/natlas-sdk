"""Finding and downloading quantised N-ATLAS weights."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import httpx

# Community GGUF conversion of NCAIR1/N-ATLaS. The official repo ships only
# safetensors, which need a CUDA GPU and ~16 GB to run unquantised.
GGUF_REPO = "QuantFactory/N-ATLaS-GGUF"
QUANTS = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
DEFAULT_QUANT = "Q4_K_M"


def cache_dir() -> Path:
    return Path(os.getenv("NATLAS_CACHE", Path.home() / ".cache" / "natlas"))


def gguf_filename(quant: str) -> str:
    if quant not in QUANTS:
        raise ValueError(f"Unknown quantisation {quant!r}. Choose from {', '.join(QUANTS)}")
    return f"N-ATLaS.{quant}.gguf"


def default_model_path(quant: str = DEFAULT_QUANT) -> Path:
    env = os.getenv("NATLAS_MODEL_PATH")
    return Path(env).expanduser() if env else cache_dir() / gguf_filename(quant)


def download(quant: str = DEFAULT_QUANT, dest: Path | None = None) -> Path:
    """Download a GGUF build, resuming a partial download if one exists."""
    dest = dest or cache_dir() / gguf_filename(quant)
    if dest.exists() and dest.stat().st_size > 0 and not dest.with_suffix(".part").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(".part")
    url = f"https://huggingface.co/{GGUF_REPO}/resolve/main/{gguf_filename(quant)}"
    done = part.stat().st_size if part.exists() else 0
    headers = {"Range": f"bytes={done}-"} if done else {}
    with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as resp:
        resp.raise_for_status()
        total = done + int(resp.headers.get("content-length", 0))
        with part.open("ab" if resp.status_code == 206 else "wb") as f:
            if resp.status_code != 206:
                done = 0
            for block in resp.iter_bytes(1 << 20):
                f.write(block)
                done += len(block)
                if total:
                    sys.stderr.write(f"\r{dest.name}: {done / total:6.1%} of {total / 1e9:.1f} GB")
    sys.stderr.write("\n")
    part.rename(dest)
    return dest
