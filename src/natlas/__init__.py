"""Python SDK for N-ATLAS, Nigeria's open multilingual language model.

Importing the package loads the HTTP clients only. The local server, speech
recognition and benchmarks live in submodules with their own optional extras.
"""

from .client import AsyncNAtlas, NAtlas
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
    StructuredOutputError,
    ToolLoopError,
)
from .languages import LANGUAGES, Language, get_language
from .tools import function_tool
from .types import ChatResponse, ToolCall, Usage

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
    "StructuredOutputError",
    "ToolCall",
    "ToolLoopError",
    "Usage",
    "function_tool",
    "get_language",
]
