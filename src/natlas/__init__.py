"""Python SDK for N-ATLAS, Nigeria's open multilingual language model.

Importing the package loads the HTTP clients only. The local server, speech
recognition and benchmarks live in submodules with their own optional extras.
"""

from .client import AsyncNAtlas, ChatResponse, NAtlas
from .errors import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NAtlasError,
    NotFoundError,
    RateLimitError,
)
from .languages import LANGUAGES, Language, get_language

__version__ = "0.1.0"

__all__ = [
    "LANGUAGES",
    "APIConnectionError",
    "APIStatusError",
    "APITimeoutError",
    "AsyncNAtlas",
    "AuthenticationError",
    "BadRequestError",
    "ChatResponse",
    "InternalServerError",
    "Language",
    "NAtlas",
    "NAtlasError",
    "NotFoundError",
    "RateLimitError",
    "get_language",
]
