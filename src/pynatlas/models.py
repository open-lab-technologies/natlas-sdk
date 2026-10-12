"""Finding and downloading quantised N-ATLAS weights."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path

import httpx

from .settings import Settings

# Community GGUF conversion of NCAIR1/N-ATLaS. The official repo ships only
# safetensors, which need a CUDA GPU and ~16 GB to run unquantised. Pinned to a
# revision so an upstream re-upload cannot change what users get without a release.
GGUF_REPO = "QuantFactory/N-ATLaS-GGUF"
GGUF_REVISION = "fdc9567e20a63e8d27736d640cabe61e937befa8"
QUANTS = ("Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0")
DEFAULT_QUANT = "Q4_K_M"


class DownloadError(Exception):
    pass


def cache_dir() -> Path:
    return Settings.from_env().cache_dir


def gguf_filename(quant: str) -> str:
    if quant not in QUANTS:
        raise ValueError(f"Unknown quantisation {quant!r}. Choose from {', '.join(QUANTS)}")
    return f"N-ATLaS.{quant}.gguf"


def default_model_path(quant: str = DEFAULT_QUANT) -> Path:
    settings = Settings.from_env()
    return settings.model_path or settings.cache_dir / gguf_filename(quant)


def expected_file(filename: str) -> tuple[str, int]:
    """SHA-256 and size Hugging Face records for `filename` at the pinned revision."""
    url = f"https://huggingface.co/api/models/{GGUF_REPO}/tree/{GGUF_REVISION}"
    resp = httpx.get(url, timeout=30, follow_redirects=True)
    resp.raise_for_status()
    for entry in resp.json():
        if entry.get("path") == filename and entry.get("lfs"):
            return entry["lfs"]["oid"], entry["lfs"]["size"]
    raise DownloadError(f"{filename} not found in {GGUF_REPO}@{GGUF_REVISION[:8]}")


def download(
    quant: str = DEFAULT_QUANT,
    dest: Path | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> Path:
    """Download a GGUF build, resuming a partial download, and verify its SHA-256.

    `progress` is called with (bytes done, total bytes) as blocks arrive.
    """
    filename = gguf_filename(quant)
    dest = dest or cache_dir() / filename
    part = dest.with_suffix(".part")
    if dest.exists() and not part.exists():
        return dest
    sha256, size = expected_file(filename)
    dest.parent.mkdir(parents=True, exist_ok=True)

    digest = hashlib.sha256()
    done = part.stat().st_size if part.exists() else 0
    if done > size:
        part.unlink()
        done = 0
    if done:
        with part.open("rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                digest.update(block)

    if done < size:
        url = f"https://huggingface.co/{GGUF_REPO}/resolve/{GGUF_REVISION}/{filename}"
        headers = {"Range": f"bytes={done}-"} if done else {}
        with httpx.stream("GET", url, headers=headers, follow_redirects=True, timeout=60) as resp:
            resp.raise_for_status()
            if resp.status_code != 206 and done:
                # Server ignored the range request: start again from zero.
                digest, done = hashlib.sha256(), 0
            with part.open("ab" if done else "wb") as f:
                for block in resp.iter_bytes(1 << 20):
                    f.write(block)
                    digest.update(block)
                    done += len(block)
                    if progress:
                        progress(done, size)

    if done != size or digest.hexdigest() != sha256:
        part.unlink(missing_ok=True)
        raise DownloadError(
            f"{filename} failed verification ({done} of {size} bytes, sha256 "
            f"{digest.hexdigest()[:12]} expected {sha256[:12]}); the partial file was "
            "removed, run the download again"
        )
    part.rename(dest)
    return dest
