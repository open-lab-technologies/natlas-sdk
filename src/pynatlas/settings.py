"""Configuration from environment variables, validated once and typed.

Every variable is optional. A malformed one fails at construction with the
variable's name, instead of surfacing later as a confusing connection error.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from .errors import NAtlasError

DEFAULT_BASE_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "n-atlas"
DEFAULT_TIMEOUT = 120.0
DEFAULT_MAX_RETRIES = 2


class ConfigError(NAtlasError, ValueError):
    """A setting is malformed. The message names the variable."""


def _base_url(value: str, source: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ConfigError(f"{source} must be an http(s) URL such as {DEFAULT_BASE_URL}, "
                          f"got {value!r}")
    return value.rstrip("/")


def _positive_float(value: str, source: str) -> float:
    try:
        number = float(value)
    except ValueError:
        raise ConfigError(f"{source} must be a number of seconds, got {value!r}") from None
    if number <= 0:
        raise ConfigError(f"{source} must be greater than 0, got {value!r}")
    return number


def _non_negative_int(value: str, source: str) -> int:
    try:
        number = int(value)
    except ValueError:
        raise ConfigError(f"{source} must be a whole number, got {value!r}") from None
    if number < 0:
        raise ConfigError(f"{source} must be 0 or more, got {value!r}")
    return number


@dataclass(frozen=True)
class Settings:
    base_url: str = DEFAULT_BASE_URL
    api_key: str | None = field(default=None, repr=False)
    model: str = DEFAULT_MODEL
    timeout: float = DEFAULT_TIMEOUT
    max_retries: int = DEFAULT_MAX_RETRIES
    model_path: Path | None = None
    cache_dir: Path = field(default_factory=lambda: Path.home() / ".cache" / "pynatlas")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        """Read NATLAS_* variables; unset or empty ones keep their defaults."""
        env = os.environ if env is None else env

        def get(name: str) -> str | None:
            value = env.get(name, "").strip()
            return value or None

        values: dict = {}
        if (v := get("NATLAS_BASE_URL")) is not None:
            values["base_url"] = _base_url(v, "NATLAS_BASE_URL")
        if (v := get("NATLAS_API_KEY")) is not None:
            values["api_key"] = v
        if (v := get("NATLAS_MODEL")) is not None:
            values["model"] = v
        if (v := get("NATLAS_TIMEOUT")) is not None:
            values["timeout"] = _positive_float(v, "NATLAS_TIMEOUT")
        if (v := get("NATLAS_MAX_RETRIES")) is not None:
            values["max_retries"] = _non_negative_int(v, "NATLAS_MAX_RETRIES")
        if (v := get("NATLAS_MODEL_PATH")) is not None:
            values["model_path"] = Path(v).expanduser()
        if (v := get("NATLAS_CACHE")) is not None:
            values["cache_dir"] = Path(v).expanduser()
        return cls(**values)

    def describe(self) -> dict[str, str]:
        """Resolved values for display, with the API key masked."""
        key = "not set" if not self.api_key else f"set ({len(self.api_key)} characters)"
        return {
            "NATLAS_BASE_URL": self.base_url,
            "NATLAS_API_KEY": key,
            "NATLAS_MODEL": self.model,
            "NATLAS_TIMEOUT": f"{self.timeout:g}",
            "NATLAS_MAX_RETRIES": str(self.max_retries),
            "NATLAS_MODEL_PATH": str(self.model_path) if self.model_path else "not set",
            "NATLAS_CACHE": str(self.cache_dir),
        }


def validate_base_url(value: str) -> str:
    return _base_url(value, "base_url")
