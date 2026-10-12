"""Imports for optional features, with an install hint when the extra is missing."""

from __future__ import annotations

import importlib
from types import ModuleType


class MissingExtraError(ImportError):
    pass


def require(module: str, extra: str) -> ModuleType:
    try:
        return importlib.import_module(module)
    except ImportError as e:
        raise MissingExtraError(
            f"{module} is needed for this feature. Install it with: pip install 'pynatlas[{extra}]'"
        ) from e
