"""Python SDK for N-ATLAS, Nigeria's open multilingual language model."""

from .client import ChatResponse, NAtlas, NAtlasError
from .languages import LANGUAGES, Language, get_language

__all__ = ["LANGUAGES", "ChatResponse", "Language", "NAtlas", "NAtlasError", "get_language"]
__version__ = "0.1.0"
